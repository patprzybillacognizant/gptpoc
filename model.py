import abc
import decimal
import itertools
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Set, Tuple

import requests
from dateutil import tz
from facebook_business.exceptions import FacebookRequestError
from facebook_business.adobjects.customaudience import CustomAudience as FBCA
from simple_history.models import HistoricalRecords
from django_fsm import FSMIntegerField, transition

from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.db.models import (
    CASCADE,
    SET_NULL,
    BooleanField,
    CharField,
    Count,
    DateTimeField,
    F,
    FloatField,
    ForeignKey,
    IntegerField,
    JSONField,
    Manager,
    Model,
    NullBooleanField,
    OneToOneField,
    PositiveBigIntegerField,
    Q,
    QuerySet,
    TextChoices,
    TextField,
    ManyToManyField,
)
from django.utils import timezone

from shadowfax.hyphen.models import (
    Retailer,
    RetailerLocation,
    StoreProductData,
    UnifiedProduct,
)
from shadowfax.truestock.mixins import FacebookTargetingMixin
from shadowfax.users.models import User
from shadowfax.utils.log import Log
from shadowfax.utils.orm import common_round
from shadowfax.utils.snowflake import query_zip_expander_location_data

log = Log()

SLACK_WEBHOOK_URL = settings.TRUESTOCK_SLACK_WEBHOOK_URL

PostalCode = str
ProductSku = str


def create_products_stock_status() -> Dict[ProductSku, bool]:
    return {}


@dataclass
class BidData:
    postal_code: PostalCode
    products_stock_status: Dict[ProductSku, bool] = field(
        default_factory=create_products_stock_status
    )
    bid_adjustment: decimal.Decimal = None
    oos_count: int = None
    weighted_oos_count: int = None


BidDataByPostalCode = Dict[PostalCode, BidData]


class BaseModel(Model):
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class BaseObjectModel(BaseModel):
    raw_data = JSONField(default=dict)

    def save(self, *args, **kwargs):
        super(BaseObjectModel, self).save(*args, **kwargs)

    class Meta:
        abstract = True


class TtdCampaign(BaseObjectModel):
    name = CharField(max_length=100)
    campaign_id = CharField(max_length=100, unique=True)
    availability = TextField()
    history = HistoricalRecords(table_name="api_historicalttdcampaign")

    resource = CharField(max_length=35, default="TRADE_DESK")

    @staticmethod
    def consolidate_geosegment_ids(adgroups) -> Set[str]:
        campaign_geo_segment_ids = set()

        for adgroup in adgroups:
            adgroup_target_geo_segment_ids = list(
                adgroup.fetch_targeted_geo_segment_ids()
            )
            campaign_geo_segment_ids.update(adgroup_target_geo_segment_ids)

        return campaign_geo_segment_ids

    def __str__(self):
        return self.name


class WeightRuleset(BaseModel):
    WEIGHT_MAP = {
        "heavy": {
            "5": [50, 30, 10, 5, 5],
            "4": [50, 30, 10, 10],
            "3": [60, 30, 10],
            "2": [70, 30],
            "1": [100],
        },
        "moderate": {
            "5": [30, 30, 20, 10, 10],
            "4": [35, 30, 20, 15],
            "3": [40, 35, 25],
            "2": [60, 40],
            "1": [100],
        },
        "equal": {
            "5": [20, 20, 20, 20, 20],
            "4": [25, 25, 25, 25],
            "3": [33.3, 33.3, 33.3],
            "2": [50, 50],
            "1": [100],
        },
    }

    name = CharField(max_length=100)
    description = TextField(
        null=True, blank=True, help_text="Optional description for weight ruleset"
    )

    overrides = ArrayField(
        FloatField(), default=list, help_text="User defined overrides to weight ruleset"
    )
    formula = ArrayField(
        FloatField(),
        default=list,
        help_text="Array of polynomial terms ordered by degree that generates curve representing weight ruleset",
    )

    version = CharField(max_length=100, help_text="weight ruleset version")

    @classmethod
    def generate_weights(cls, weight_type, num_weights):
        return cls.WEIGHT_MAP[weight_type][str(num_weights)]

    def __str__(self):
        return f"{self.name} v{self.version}"


class OptimizerStateEnum(object):
    READY = 0
    FAILED = 11


class OptimizerType(BaseModel):
    OUT_OF_STOCK = "out_of_stock"
    CONQUEST = "conquest"

    class JSONAPIMeta:
        resource_name = "optimizer_types"

    name = TextField()
    slug = TextField(unique=True)


class Optimizer(BaseModel):
    DEFAULT_BID_ADJUSTMENTS = {
        "1": 0.2,
        "2": 0.4,
        "3": 0.6,
        "4": 0.8,
        "5": 1.0,
    }  # TODO - Make dynamic based on product count https://bvaccel.atlassian.net/browse/HYP-505

    class CadenceOptions(TextChoices):
        HOURLY = "H", "Hourly"
        DAILY = "D", "Daily"
        WEEKLY = "W", "Weekly"

    class MarketingPlatforms(TextChoices):
        THE_TRADE_DESK = "TTD", "The Trade Desk"
        FACEBOOK = "FBK", "Facebook"
        WALMART_DSP = "WMT", "WalMart DSP"

    class JSONAPIMeta:
        resource_name = "optimizers"

    name = TextField()
    optimizer_type = ForeignKey(
        OptimizerType,
        on_delete=SET_NULL,
        null=True,
    )

    hero_products = ArrayField(CharField(max_length=16), default=list, blank=True)
    weight = CharField(max_length=100, default="equal")

    conquest_hero_product = ForeignKey(UnifiedProduct, null=True, on_delete=SET_NULL)
    conquest_products = ArrayField(TextField(), default=list, blank=True)

    active = BooleanField(default=True)
    state = FSMIntegerField(default=OptimizerStateEnum.READY, protected=True)
    update_frequency = CharField(
        max_length=1, choices=CadenceOptions.choices, default=CadenceOptions.DAILY
    )
    adjustments = JSONField(default=DEFAULT_BID_ADJUSTMENTS, blank=True)
    last_completed_run = DateTimeField(null=True, default=None, blank=True)
    last_attempted_run = DateTimeField(null=True, default=None, blank=True)

    created_by = ForeignKey(User, related_name="+", null=True, on_delete=SET_NULL)
    updated_by = ForeignKey(User, related_name="+", null=True, on_delete=SET_NULL)

    history = HistoricalRecords(table_name="api_historicaloptimizer")

    retailer = ForeignKey(
        Retailer, null=True, blank=True, on_delete=SET_NULL, related_name="optimizers"
    )

    platform = CharField(
        max_length=3,
        choices=MarketingPlatforms.choices,
        default=MarketingPlatforms.THE_TRADE_DESK,
        null=True,
        blank=True,
    )

    meta = JSONField(default=dict, blank=True)

    @property
    def freshness_threshold(self) -> int:
        """
        Returns an integer representing how many hours old product data can be before this optimizer
        considers it "stale"
        """
        hour = 1
        match self.update_frequency:
            case self.CadenceOptions.HOURLY:
                return hour
            case self.CadenceOptions.DAILY:
                return hour * 24
            case self.CadenceOptions.WEEKLY:
                return hour * 24 * 7

        return 0

    def fetch_target_audience_template(self):
        """
        Get or create a target-audience template to stamp multiple PUT payloads on platform APIs
        e.g. (TTD Bid Lists, Facebook Adsets, etc.)
        """

        try:
            return OptimizedTargetAudience.objects.get(optimizer=self)
        except OptimizedTargetAudience.DoesNotExist:
            return OptimizedTargetAudience.objects.create(
                optimizer=self, name=f"Truestock - Optimizer: {self.pk}"
            )

    def __str__(self):
        return self.name

    @transition(field=state, source="*", target=OptimizerStateEnum.FAILED)
    def handle_optimizer_errors(self, error, task_id: str = None):
        log.error(f"[ERROR] {error}")

        meta = self.meta or dict()
        failure = {
            "details": str(error),
            "timestamp": datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M"),
        }

        if meta.get("failures") is not None:
            meta["failures"].append(failure)
        else:
            meta["failures"] = [failure]

        self.meta = meta
        self.send_slack_notification("error", {"error": error}, task_id=task_id)

    @transition(field=state, source="*", target=OptimizerStateEnum.READY)
    def reset_to_ready(self):
        """
        Resets the optimizer to state 'READY'. Deprecated
        """
        pass

    # Workaround for django-fsm conflict with Django's refresh_from_db ORM method
    # https://github.com/viewflow/django-fsm/issues/89#issuecomment-797532572
    def pytest_refresh_from_db(self, using=None, fields=None):
        """Reload instance from database."""
        protected = []
        for field in self._meta.get_fields():
            if getattr(field, "protected", False):
                protected.append(field)
                field.protected = False

        super().refresh_from_db(using=using, fields=fields)
        for field in protected:
            field.protected = True

    def get_all_product_ids(self) -> List[str]:
        optimizer_product_ids = [
            p
            for p in (
                getattr(self.conquest_hero_product, "product_id", None),
                *self.hero_products,
                *self.conquest_products,
            )
            if p is not None
        ]
        return optimizer_product_ids

    @staticmethod
    def query_geosegments(campaign_geo_segment_ids: Iterable) -> "QuerySet[GeoSegment]":
        """
        When collecting geosegments for a campaign, there's no indication of geosegment type. Since we're only after
        zipcode geosegments, we first reference our GeoSegment table against campaign geosegments to filter the results
        down (GeoSegment records are strictly zipcode types)
        """

        geo_segments = GeoSegment.objects.filter(
            geo_segment_id__in=campaign_geo_segment_ids
        )

        if not geo_segments:

            class GeoSegmentException(Exception):
                pass

            raise GeoSegmentException(
                "No zip-level GeoSegments found for the given IDs"
            )

        return geo_segments

    def needs_to_run(self) -> bool:
        """
        Checks to see if the optimizer needs to run based on it's last completion/attempt.
        Daily/Weekly runs are "checkpointed" to 12pm Central Time
        Failed optimizers will not retry before the cooldown period is over (1 hour).
        """
        timedelta_map = {
            self.CadenceOptions.HOURLY[0]: timedelta(hours=1),
            self.CadenceOptions.DAILY[0]: timedelta(days=1),
            self.CadenceOptions.WEEKLY[0]: timedelta(weeks=1),
        }
        central_tz = tz.gettz("America/Chicago")
        now = datetime.now(tz=central_tz)

        last_completed_checkpoint = (self.last_completed_run or now).astimezone(
            central_tz
        )
        if self.update_frequency != self.CadenceOptions.HOURLY[0]:
            last_completed_checkpoint = last_completed_checkpoint.replace(
                hour=12, minute=0
            )

        time_since_last_completion = now - last_completed_checkpoint
        interval_timedelta = timedelta_map[self.update_frequency]
        never_ran = self.last_completed_run is None

        cooling_down = self.last_attempted_run and (
            now - self.last_attempted_run
        ) < timedelta(hours=1)
        optimizer_needs_to_run = (
            never_ran or time_since_last_completion > interval_timedelta
        )
        return optimizer_needs_to_run and not cooling_down and self.active

    @staticmethod
    def get_retailer_locations_within_campaign_targeting(
        retailer_name: str, postal_codes: Iterable[str]
    ):
        locations = RetailerLocation.objects.select_related("retailer").filter(
            retailer__name=retailer_name
        )
        if postal_codes:
            locations = locations.filter(postal_code__in=postal_codes)
        if postal_codes and not locations:
            log.warn(
                f"No locations found for {len(list(postal_codes))} postal_codes provided"
            )  # this should halt and throw. Passing an empty list will only create problems later
        return locations.values_list("id", "location_id", "postal_code")

    @staticmethod
    def retrieve_and_process_zip_expander_data(
        retailer_name, store_ids, targeted_postal_codes
    ):
        zip_expander_data = query_zip_expander_location_data(
            retailer_name, store_ids, targeted_postal_codes
        )

        for z in zip_expander_data:
            z["postal_code"] = str(z.get("postal_code")).zfill(5)
            z["store_id"] = str(z.get("store_id"))

        return zip_expander_data

    @staticmethod
    def join_zip_expander_data_in_batches_chunked_by_product(
        product_id: str,
        spd_for_targeted_products: List[dict],
        zip_expander_data: List[dict],
    ):
        spd_for_single_product = [
            s for s in spd_for_targeted_products if s.get("product_id") == product_id
        ]
        expanded_store_data = spd_for_single_product.copy()

        for data in spd_for_single_product:
            store_id = data.get("store_id")
            in_stock = data.get("in_stock")

            zips_within_proximity = [
                {**z, "in_stock": in_stock, "product_id": product_id}
                for z in zip_expander_data
                if str(z.get("store_id")) == store_id
            ]
            expanded_store_data = [*expanded_store_data, *zips_within_proximity]

        return expanded_store_data

    def merge_together_zip_expander_and_store_product_data(
        self,
        retailer_name: str,
        product_ids: List[str],
        store_ids: List[str],
        spd_for_all_targeted_products: List[dict],
        targeted_postal_codes: List[str],
    ):

        zip_expander_data = self.retrieve_and_process_zip_expander_data(
            retailer_name, store_ids, targeted_postal_codes
        )

        expanded_store_level_product_data: list[dict] = list()
        for p in product_ids:
            batch_result = self.join_zip_expander_data_in_batches_chunked_by_product(
                p, spd_for_all_targeted_products, zip_expander_data
            )
            expanded_store_level_product_data = [
                *expanded_store_level_product_data,
                *batch_result,
            ]

        return expanded_store_level_product_data

    def get_average_of_product_availability_for_all_locations_within_geolocation(
        self, store_product_data
    ):
        stock_averages_for_product = {}
        record_count = 0
        stock_aggregate = 0
        current_key = ""

        sorted_stored_level_product_data = (
            self.sort_expanded_store_level_product_data_based_on_platform(
                store_product_data
            )
        )

        # 99 Total Items
        for index, p in enumerate(sorted_stored_level_product_data):
            product_id = p.get("product_id")
            # Note: p["in_stock"]: bool | None
            in_stock = int(bool(p.get("in_stock")))

            geo_key = (
                p.get("city_id") if self.platform == "FBK" else p.get("postal_code")
            )
            new_key = f"{geo_key}-{product_id}"

            if geo_key not in stock_averages_for_product:
                stock_averages_for_product[geo_key] = {}

            if current_key not in [new_key, ""]:
                average = round(stock_aggregate / record_count, 2)

                prev_geo_key = current_key.split("-")[0]
                prev_product_id = current_key.split("-")[1]
                stock_averages_for_product[prev_geo_key][prev_product_id] = average

                record_count = 1
                stock_aggregate = in_stock
            else:
                record_count += 1
                stock_aggregate += in_stock

            # If this is the last iteration
            if (index + 1) == len(store_product_data):
                average = round(stock_aggregate / record_count, 2)
                stock_averages_for_product[geo_key][product_id] = average

            current_key = new_key

        return stock_averages_for_product

    @staticmethod
    def join_facebook_city_ids_into_spd_on_postal_code(store_product_data):
        """
        Translates postal codes into FB cities
        """
        postal_codes_with_facebook_city_ids = (
            FacebookPostalCodeGeolocationEntity.objects.select_related("primary_city")
            .exclude(primary_city__city_id__icontains="us")
            .values_list("name", "primary_city__city_id")
        )

        if not postal_codes_with_facebook_city_ids:
            log.warn("postal_codes_with_facebook_city_ids: empty array!")

        fb_city_id_zip_map = {k: v for (k, v) in postal_codes_with_facebook_city_ids}

        spd = list()

        for record in store_product_data:
            postal_code = record.get("postal_code")
            if (city_id := fb_city_id_zip_map.get(postal_code)) is not None:
                record["city_id"] = city_id
                spd.append(record)
            else:
                log.warn(
                    f"join_facebook_city_ids_into_spd_on_postal_code: No Facebook city id found for postal code {postal_code}"
                )

        if not spd:
            log.warn(
                "join_facebook_city_ids_into_spd_on_postal_code: No Facebook city ids found for postal codes in store_product_data"
            )

        return spd

    @staticmethod
    def populate_defaults_of_zero_for_missing_store_product_data(
        store_ids, product_ids, store_product_data, store_to_zip_map
    ):
        """
        We won't always have store-level data available for all optimizer products at every location.
        For any product/store combination we don't have store-level data for, we insert a 0 value for in_stock.

        To find out what product/store data is missing we get the difference of two lists, a list of all possible
        store_id/product_id combinations, and a unique list of store_id/product_id combinations returns from a db
        query.

        __________________________________________________________________________________
        store_ids = [111, 222]
        product_ids = [12345, 34567]
        store_product_pairs_available = [(111, 12345), (111, 34567), (222, 12345), (222, 34567)]

        store_product_data_available = [(111, 12345), (111, 34567), (222, 12345)]
        store_product_pairs_possible.difference(store_product_pairs_present)
        >> [(222, 34567)]
        """

        store_product_pairs_possible = set(
            [pair for pair in itertools.product(store_ids, product_ids)]
        )
        store_product_pairs_present = set(
            [(r.get("store_id"), r.get("product_id")) for r in store_product_data]
        )

        missing_store_level_data = store_product_pairs_possible.difference(
            store_product_pairs_present
        )
        for missing_data in missing_store_level_data:
            store_id, product_id = missing_data

            store_product_data.append(
                {
                    "store_id": store_id,
                    "postal_code": store_to_zip_map[store_id],
                    "product_id": product_id,
                    "in_stock": 0,
                }
            )

    def get_store_level_product_data_for_locations_within_targeting(
        self, product_ids: List[str], targeted_postal_codes: Iterable[str] = None
    ):

        retailer_name = self.retailer.name
        # targeted_postal_codes is the zip list from optimize_audience_targeting -> run_inputs_through_truestock
        retailer_locations = self.get_retailer_locations_within_campaign_targeting(
            retailer_name, targeted_postal_codes
        )

        if targeted_postal_codes and product_ids and not retailer_locations:
            log.warn(
                f"LOOKUP FAIL for retailer {retailer_name} and {len(list(targeted_postal_codes))} postal codes"
            )

        location_ids = [location[0] for location in retailer_locations]
        store_ids = [location[1] for location in retailer_locations]
        postal_codes = [location[2] for location in retailer_locations]
        store_to_zip_map = {k: v for k, v in zip(store_ids, postal_codes)}

        spd = list(
            StoreProductData.objects.select_related("retailer_location")
            .filter(
                retailer_location_id__in=location_ids,
                product_id__in=product_ids,
                updated_at__gte=datetime.utcnow() - timedelta(hours=24),
            )
            .annotate(postal_code=F("retailer_location__postal_code"))
            .values("in_stock", "postal_code", "store_id", "product_id")
        )

        if not spd:
            log.warn(
                f"No store-level data found for {len(location_ids)} locations and {len(product_ids)} products"
            )  # this should probably halt: there are no known products here

        # This has to be done 'before' merging zip-expander data into store-level product data
        self.populate_defaults_of_zero_for_missing_store_product_data(
            store_ids, product_ids, spd, store_to_zip_map
        )

        store_level_product_data = (
            self.merge_together_zip_expander_and_store_product_data(
                retailer_name, product_ids, store_ids, spd, list(targeted_postal_codes)
            )
        )

        if self.platform == "FBK":
            store_level_product_data = (
                self.join_facebook_city_ids_into_spd_on_postal_code(
                    store_level_product_data
                )
            )

        return store_level_product_data

    def sort_expanded_store_level_product_data_based_on_platform(self, data):
        """
        Performs a nested search on store-level product data objects based on the platform
        """
        if self.platform == "FBK":
            return sorted(
                data, key=lambda d: (d["city_id"], d["postal_code"], d["product_id"])
            )
        else:
            return sorted(data, key=lambda d: (d["postal_code"], d["product_id"]))

    def get_stock_averages_for_products_at_all_targeted_stores(
        self, product_ids: List[str], postal_codes: Iterable[str] = None
    ):
        store_level_product_data = (
            self.get_store_level_product_data_for_locations_within_targeting(
                product_ids, postal_codes
            )
        )

        return self.get_average_of_product_availability_for_all_locations_within_geolocation(
            store_level_product_data
        )

    def use_stock_averages_to_calculate_bid_adjustments(self, product_stock_averages):
        bid_data_by_postal = {}  # type: BidDataByPostalCode
        for postal_code, products_stock_status in product_stock_averages.items():
            bid_data = BidData(
                postal_code=postal_code, products_stock_status=products_stock_status
            )
            bid_data_weighted_and_adjusted = (
                self.calculate_bid_adjustments_for_bid_data(bid_data)
            )
            bid_data_by_postal[postal_code] = bid_data_weighted_and_adjusted

        return bid_data_by_postal

    def populate_bid_data_by_postal_code(
        self, product_ids: List[str], postal_codes: Iterable[str] = None
    ):
        stock_averages = self.get_stock_averages_for_products_at_all_targeted_stores(
            product_ids, postal_codes
        )

        bid_adjustments = self.use_stock_averages_to_calculate_bid_adjustments(
            stock_averages
        )

        return bid_adjustments

    @staticmethod
    def convert_bid_data_by_postal_code_to_oos_zips_by_product_id(
        bid_data_by_postal_code: BidDataByPostalCode,
    ) -> Dict[str, list]:
        oos_zips_by_product_id = defaultdict(list)
        for z, d in bid_data_by_postal_code.items():
            for product_id, stock_status in d.products_stock_status.items():
                if stock_status == False:
                    oos_zips_by_product_id[product_id].append(z)
        return oos_zips_by_product_id

    def determine_oos_counts(self, products_stock_status, products):
        product_weight_map = {p: i for (i, p) in enumerate(products)}
        weights = WeightRuleset.generate_weights(self.weight, len(products))

        oos_weight_totals = decimal.Decimal(0)
        for product_id, product_stock_average in products_stock_status.items():
            product_index = product_weight_map[product_id]
            product_weight = decimal.Decimal(weights[product_index]) / decimal.Decimal(
                sum(weights)
            )

            if product_stock_average < 1:
                oos_weight_totals += decimal.Decimal(
                    product_weight * decimal.Decimal(1 - product_stock_average)
                )

        return (
            int(common_round(oos_weight_totals * len(products))),
            len(products) - sum(products_stock_status.values()),
        )

    def calculate_bid_adjustments_for_bid_data(
        self,
        bid_data: BidData,
    ) -> BidData:
        opt_type = self.optimizer_type.slug

        optimizer_products = (
            self.hero_products
            if opt_type == OptimizerType.OUT_OF_STOCK
            else self.conquest_products
        )
        products_stock_status = bid_data.products_stock_status

        if opt_type == OptimizerType.CONQUEST:
            hero_product_id = self.conquest_hero_product.product_id
            hero_product_in_stock = products_stock_status.pop(hero_product_id)

            weighted_oos_count, oos_count = self.determine_oos_counts(
                products_stock_status, optimizer_products
            )
            bid_adjustment = (
                1.0 + self.adjustments.get(str(weighted_oos_count), 0.0)
                if hero_product_in_stock
                else 0.0
            )

            # Re-insert conquest hero product now that the math is done
            products_stock_status[hero_product_id] = hero_product_in_stock

        else:
            weighted_oos_count, oos_count = self.determine_oos_counts(
                products_stock_status, optimizer_products
            )
            bid_adjustment = (
                round(1.0 - self.adjustments[str(weighted_oos_count)], 2)
                if weighted_oos_count
                else 1.0
            )

        bid_data.bid_adjustment = bid_adjustment
        bid_data.oos_count = round(oos_count, 2)
        bid_data.weighted_oos_count = weighted_oos_count

        return bid_data

    def check_known_distribution_for_products(
        self,
        retailer: Retailer = None,
        product_ids: List = None,
        postal_codes: List = None,
        threshold: float = 0.90,
    ) -> Tuple[bool, List[Dict[str, float]]]:
        if not retailer:
            retailer = self.retailer
        if not product_ids:
            product_ids = self.get_all_product_ids()

        match self.update_frequency:
            case self.CadenceOptions.HOURLY:
                cadence = 1
            case self.CadenceOptions.DAILY:
                cadence = 24
            case self.CadenceOptions.WEEKLY:
                cadence = 24 * 7
            case _:
                cadence = 24

        location_query = {}  # type: Dict[str, Any]
        location_query["retailer"] = retailer
        store_product_data_query = {
            "product_id__in": product_ids,
            "unified_product__retailer": retailer,
            "updated_at__gte": datetime.utcnow() - timedelta(hours=cadence),
        }
        if postal_codes is not None:
            location_query["postal_code__in"] = postal_codes
            store_product_data_query[
                "retailer_location__postal_code__in"
            ] = postal_codes

        location_count = RetailerLocation.objects.filter(**location_query).count()
        product_data_agg = list(
            StoreProductData.objects.select_related(
                "unified_product__retailer", "retailer_location"
            )
            .filter(**store_product_data_query)
            .values("unified_product__product_id")
            .annotate(
                checked_distribution=(
                    Count("unified_product__product_id") / float(location_count)
                )
            )
        )

        known_distribution_is_adequate = bool(
            product_data_agg
            and all(
                agg["checked_distribution"] >= threshold for agg in product_data_agg
            )
        )
        return known_distribution_is_adequate, product_data_agg

    def aggregate_oos_counts(
        self, bid_data_by_postal: BidDataByPostalCode
    ) -> Dict[str, int]:
        oos_counts = Counter(str(b.oos_count) for b in bid_data_by_postal.values())
        aggregate_counts = dict(oos_counts)
        if self.optimizer_type.slug == OptimizerType.CONQUEST:
            conquest_hero_product_id = self.conquest_hero_product.product_id
            aggregate_counts["conquest_hero_product_oos_count"] = sum(
                [
                    not bid_data.products_stock_status[conquest_hero_product_id]
                    for bid_data in bid_data_by_postal.values()
                ]
            )
        return aggregate_counts

    ################################
    #    DJANGO FSM TRANSITIONS    #
    ################################
    # All roads lead back to READY, be noisy and keep retrying regardless of failure
    @transition(field=state, source="*", target=OptimizerStateEnum.READY)
    def optimize_audience_targeting(self, platform, task_id: str = None):
        log.info(f"Optimizing audience targeting for optimizer: {self.id}")

        # Inputs required to perform internal logic, e.g. Pre-existing Tradedesk Geosegments on TTD campaigns
        inputs = platform.obtain_required_inputs()

        # Uses inputs to query store-level product data and generate bid adjustments
        bid_data_by_postal_code = platform.run_inputs_through_truestock(inputs)

        # update optimizer with bid adjustments
        platform.update_target_audience_template(bid_data_by_postal_code)

        # Interfaces with ad-management platform's APIs to apply Truestock results
        platform.update_target_audience_on_platform(task_id)

    @staticmethod
    def format_distribution_agg_message(dist_agg) -> str:
        formatted_aggs = []

        for dist_data in dist_agg:
            product_id = dist_data["unified_product__product_id"]
            checked_distribution = dist_data["checked_distribution"]
            formatted_distribution = f"{round(checked_distribution * 100, 1)}%"
            formatted_aggs.append((product_id, formatted_distribution))

        agg_message = "\n".join(
            [f"{product_id}: {dist}" for (product_id, dist) in formatted_aggs]
        )
        return agg_message

    def run(self, force=False, task_id: str = None):
        from shadowfax.ad_platforms import AD_MANAGEMENT_PLATFORMS

        platform = AD_MANAGEMENT_PLATFORMS[self.platform](optimizer=self)

        if not force and not self.needs_to_run():
            log.info("Optimizer has run recently, skipping")
            return

        valid, error_msg = platform.validate_data_on_optimizer()
        if valid is not True:
            platform._log_error(error_msg)
            return

        targeted_postal_codes = platform.targeted_postal_codes()

        now = timezone.now()
        self.last_attempted_run = now
        self.save()

        try:
            (
                known_distribution_is_adequate,
                dist_agg,
            ) = self.check_known_distribution_for_products(
                postal_codes=targeted_postal_codes
            )

            if not known_distribution_is_adequate:
                agg_message = self.format_distribution_agg_message(dist_agg)
                log.info(f"Known product distribution is inadequate: {agg_message}")
                self.send_slack_notification(
                    "error",
                    {
                        "error": dedent(
                            f"""
                            Current checked product distribution is inadequate to run optimizer:\n
                            Known distribution breakdown:\n{agg_message}
                            """
                        )
                    },
                    task_id=task_id,
                )
                return

            self.optimize_audience_targeting(platform, task_id=task_id)
            self.last_completed_run = now

        except Exception as e:
            self.handle_optimizer_errors(e, task_id=task_id)
            self.reset_to_ready()

        self.save()

    def campaign(self):
        if self.platform in {"TTD", "WMT"}:
            return self.ttd_ad_groups.first().ttd_campaign
        else:
            return self.facebook_adsets.first().facebook_campaign

    # Optimized all Ad Groups: x our of stock products
    def send_slack_notification(self, notification_type, args, task_id: str = None):
        campaign = self.campaign()
        headers = {"Content-Type": "application/json"}
        payload = {}

        base_tdd_partner_url = (
            f"https://desk.thetradedesk.com/app/partner/{settings.TTD_PARTNER_ID}"
        )
        campaign_details_url = (
            f"{base_tdd_partner_url}/buy/campaign/{campaign.campaign_id}/details"
        )

        if notification_type == "successful_optimization":
            bid_adjustment_counts = args.get("bid_adjustment_counts")
            adjusted = args.get("adjusted")
            targeted = args.get("targeted")

            bid_adjustments_str = "*Bid adjustment counts:*\n\n"
            for bid_adjustment_value, count in bid_adjustment_counts.items():
                bid_adjustments_str += f"*{bid_adjustment_value}:*   {count}\n"

            payload = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"Campaign '{campaign.name}' has been optimized! :rocket:",
                        },
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Type:*\n {self.optimizer_type.name}",
                            },
                        ],
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Platform:*\n {self.platform}",
                            },
                        ],
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Adjusted:*\n{adjusted} / {targeted}\n\n`[Adjusted zip codes] / [all targeted zip codes]`",
                            },
                        ],
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": bid_adjustments_str},
                        ],
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View Campaign",
                                    "emoji": True,
                                },
                                "style": "primary",
                                "url": campaign_details_url,
                            }
                        ],
                    },
                    {"type": "divider"},
                ]
            }
        elif notification_type == "error":
            error = args.get("error")

            payload = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"Campaign '{campaign.name}' failed to fully optimize :warning:",
                        },
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"""
                                > Optimizer `{self.name}` failed to adjust bidlists for one or more ad groups on campaign `{campaign.name}`. Optimization will be re-attempted.
                                """,
                            },
                        ],
                    },
                    {"type": "divider"},
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Error*"},
                        ],
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"> {error}"},
                        ],
                    },
                ]
            }

        else:
            payload = {"blocks": []}

        if task_id:
            payload["blocks"].append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"async job: {task_id=}"},
                    ],
                },
            )

        requests.post(SLACK_WEBHOOK_URL, headers=headers, json=payload)

    class Meta:
        ordering = ("id",)


BIDLIST_ADJUSTMENT_TYPES = (
    ("TargetList", "Target List"),
    ("BlockList", "Block List"),
    ("Optimized", "Optimized"),
)


class TtdBidList(BaseObjectModel):
    bidlist_id = CharField(max_length=100)

    has_geosegment_id = BooleanField(default=False)
    bidlist_adjustment_type = CharField(
        max_length=20,
        choices=BIDLIST_ADJUSTMENT_TYPES,
        default="TargetList",
    )
    geosegments = ArrayField(CharField(max_length=100), default=list)

    enabled = BooleanField(default=False)

    resource = CharField(max_length=35, default="TRADE_DESK")

    class Meta:
        unique_together = ("bidlist_id", "resource")


class TtdAdGroup(BaseObjectModel):
    name = CharField(max_length=100)
    ad_group_id = CharField(max_length=100, unique=True)
    history = HistoricalRecords(table_name="api_historicalttdadgroup")

    ttd_campaign = ForeignKey(
        TtdCampaign, on_delete=CASCADE, related_name="ttd_ad_groups"
    )

    optimizer = ForeignKey(
        Optimizer,
        on_delete=SET_NULL,
        related_name="ttd_ad_groups",
        null=True,
        blank=True,
    )

    resource = CharField(max_length=35, default="TRADE_DESK")

    bidlists = ManyToManyField(TtdBidList)

    class JSONAPIMeta:
        resource_name = "ttd_ad_groups"

    def status(self):
        return self.raw_data.get('Availability') == 'Available'

    def __str__(self):
        return self.name

    def fetch_targeted_geo_segment_ids(self):
        for bid_list_geosegments in self.bidlists.filter(
            has_geosegment_id=True
        ).values_list("geosegments", flat=True):
            for geosegment in bid_list_geosegments:
                yield geosegment

    def de_optimize(self):
        from shadowfax.ad_platforms import AD_MANAGEMENT_PLATFORMS

        log.info(f"Disassociating optimized bidlists from AdGroup: {self.id}")

        platform = AD_MANAGEMENT_PLATFORMS[self.optimizer.platform]()
        platform.revert_audience_targeting_optimizations(self)

    class Meta:
        unique_together = (
            "ad_group_id",
            "optimizer",
        )


class GeoSegment(BaseModel):
    geo_segment_id = CharField(max_length=100, unique=True)
    zipcode = CharField(max_length=100, unique=True)

    class Meta:
        ordering = ("id",)

    class JSONAPIMeta:
        resource_name = "geo_segments"


class OptimizedTargetAudience(BaseModel):
    """
    This model does not represent TradeDesk BidLists or Facebook audiences directly, which is why there is no 'bidlist_id' field.
    It is intended to be a BidList 'mold', generated from an Optimizer for the purpose of upserting TradeDesk bidlists onto AdGroups.

    This gives us the ability to track optimized BidList changes, without having to store Abacus results history or perform
    syncs of TradeDesk bidlists. To track history on 'all' bidlists however, a sync would still be required.
    """

    name = CharField(max_length=255)

    target = JSONField(default=dict)

    oos_aggregates = JSONField(
        default=dict,
        help_text="List of dicts each representing occurrences of an oos magnitude by location. E.g. '7 stores had 3 out of 5 products oos'",
    )
    bid_adjustment_aggregates = JSONField(
        default=dict,
        help_text="List of dicts where key is a bid-adjustment (0.2, 0.6, ..), and value is the sum of locations that got that adjustment",
    )
    num_targeted = IntegerField(
        default=0, help_text="Total count of targeted zips on ad-platform campaigns"
    )
    num_adjusted = IntegerField(default=0, help_text="Total count of adjusted zips")
    oos_zips_by_product_id = JSONField(default=dict)

    optimizer = OneToOneField(
        Optimizer,
        on_delete=SET_NULL,
        related_name="optimized_target_audience",
        null=True,
        blank=True,
    )
    associated_bidlist_ids = CharField(max_length=255, null=True, blank=True)
    associated_adset_ids = CharField(max_length=255, null=True, blank=True)

    raw_data = JSONField(default=dict)
    history = HistoricalRecords(table_name="api_historicaloptimizedtargetaudience")

    def __str__(self):
        return self.name

    def generate_request_payload(self):
        from shadowfax.ad_platforms.the_trade_desk.constants import INIT_MAP
        from shadowfax.resource_locking_client.client import Resource

        match self.optimizer.platform:
            case self.optimizer.MarketingPlatforms.THE_TRADE_DESK:
                partner_id = INIT_MAP[Resource.TRADE_DESK]["TTD_PARTNER_ID"]
            case self.optimizer.MarketingPlatforms.WALMART_DSP:
                partner_id = INIT_MAP[Resource.WALMART_DSP]["TTD_PARTNER_ID"]
            case _:
                raise NotImplementedError()

        return {
            "Name": f"Truestock - Optimizer: {self.optimizer.id}",
            "BidListType": "BidAdjustment",
            "BidListSource": "User",
            "BidlistOwner": "Partner",
            "BidlistOwnerId": partner_id,
            "BidListAdjustmentType": "Optimized",
            "BidLines": self.target,
            "ResolutionType": "ApplyAverageAdjustment",
            "BidListDimensions": ["HasGeoSegmentId"],
        }

    class JSONAPIMeta:
        resource_name = "optimized_bid_lists"


###########################
# Social (Facebook, etc.) #
###########################
# BID_STRATEGY_CHOICES = (
#     ("LOWEST_COST_WITHOUT_CAP", "Lowest Cost Without Cap"),
#     ("LOWEST_COST_WITH_BID_CAP", "Lowest Cost With Bid Cap"),
#     ("COST_CAP", "Cost Cap"),
#     ("X", "No Strategy Set"),
# )

STATUS_CHOICES = (
    ("AC", "Active"),
    ("PA", "Paused"),
    ("DE", "Deleted"),
    ("AR", "Archived"),
)
EFFECTIVE_STATUS_CHOICES = STATUS_CHOICES + (
    ("IP", "In Process"),
    ("WI", "With Issues"),
)


class FacebookCityGeolocationEntity(Model):
    city_id = CharField(max_length=100, unique=True)
    name = CharField(max_length=100)
    country_code = CharField(max_length=4)
    region = CharField(max_length=100, null=True, blank=True)
    region_id = CharField(max_length=20, null=True, blank=True)
    supports_region = BooleanField(default=True)
    supports_city = BooleanField(default=True)


class FacebookPostalCodeGeolocationEntity(Model):
    postal_code = CharField(max_length=100, unique=True)
    name = CharField(max_length=100)
    primary_city = ForeignKey(
        FacebookCityGeolocationEntity, on_delete=CASCADE, related_name="postal_codes"
    )


class FacebookMarketingCampaign(Model):
    """
    The highest-level organizational structure within a Facebook Marketing Account. This should represent a single
    objective for an advertiser: e.g., drive add-to-cart actions, drive on-page conversions, etc.
    """

    campaign_id = CharField(
        max_length=100,
        help_text="The ID of the campaign in Facebook's graph API",
        unique=True,
    )
    account_id = CharField(
        max_length=100, help_text="ID of the ad account that owns this campaign"
    )
    effective_status = CharField(max_length=2, choices=EFFECTIVE_STATUS_CHOICES)
    name = CharField(max_length=255)
    start_time = DateTimeField(null=True)
    stop_time = DateTimeField(null=True)
    status = CharField(max_length=2, choices=STATUS_CHOICES)
    updated_time = DateTimeField()
    last_sync = DateTimeField()

    def __str__(self):
        return f"FB Marketing Campaign {self.campaign_id}"


class FacebookMarketingAdSet(FacebookTargetingMixin, Model):
    adset_id = CharField(
        max_length=100,
        help_text="The ID of the ad set in Facebook's graph API",
        unique=True,
    )
    bid_amount = IntegerField(help_text="Bid cap or target cost for this ad set.")
    facebook_campaign = ForeignKey(
        FacebookMarketingCampaign,
        on_delete=CASCADE,
        related_name="facebook_adsets",
        null=False,
    )
    name = TextField()
    start_time = DateTimeField(null=True)
    stop_time = DateTimeField(null=True)
    status = CharField(max_length=2, choices=STATUS_CHOICES)
    last_sync = DateTimeField()

    optimizer = ForeignKey(
        Optimizer,
        on_delete=SET_NULL,
        related_name="facebook_adsets",
        null=True,
        blank=True,
    )

    def __str__(self):
        return f"FB AdSet {self.adset_id}"

    class JSONAPIMeta:
        resource_name = "facebook_adsets"


###
# Audience Builder Records
###
class StaleAudienceManager(Manager):
    """
    A Manager for CustomAudiences to help with interacting with "stale" audiences.
    An audience is considered "stale" if:
        - they are not archived, and
        - they haven't been synced within X days (currently 10), OR they've never been synced
    """

    def get_queryset(self):
        return (
            super()
            .get_queryset()
            .filter(
                Q(archived=False)
                & (
                    Q(last_sync__lte=timezone.now() - timedelta(days=10))
                    | Q(last_sync__isnull=True)
                )
            )
        )


class CustomAudience(BaseModel):
    last_sync = DateTimeField(
        null=True,
        default=None,
        help_text="Timestamp of the last attempt to synchronize with external platform",
    )
    last_sync_successful = NullBooleanField(
        default=None,
        help_text="True if last sync attempt was successful, False if not, None if no sync attempt has ever been made",
    )
    archived = BooleanField(
        default=False,
        help_text="True if this audience has been deleted in the external platform",
    )
    audience_id = CharField(
        max_length=255,
        null=True,
        default=None,
        help_text="Identifier of this audience in external platform",
    )

    # Universal audience creation & maintenance parameters
    name = CharField(
        max_length=255,
        null=False,
        blank=False,
        help_text="Name / Label for an audience",
    )
    conversions = BooleanField(
        default=False,
        help_text="True for conversion-only audiences, False for all views / interactions",
    )
    break_date = DateTimeField(
        null=True,
        default=None,
        blank=True,
        help_text="Only shops created on or after the break date will contribute to an audience",
    )
    rolling_days = IntegerField(
        null=True,
        blank=True,
        default=None,
        help_text="If populated, break_date will be set to (currentDate - rolling_days) on sync operations",
    )
    brand = CharField(
        max_length=255,
        null=True,
        default=None,
        blank=True,
        help_text="Brand name filter: only shops with a product with the given brand name will contribute to an audience",
    )
    retailers = ArrayField(
        CharField(max_length=255),
        null=True,
        default=None,
        blank=True,
        help_text="Retailer filter: only shops with a product from given retailers will contribute to an audience",
    )
    vertical = CharField(
        max_length=4,
        null=True,
        default=None,
        blank=True,
        help_text="Vertical / Campaign Category filter: only shops associated with a campaign in a given vertical will contribute to an audience",
    )

    # Local data tracking
    shop_ids = ArrayField(
        CharField(max_length=15),
        null=True,
        default=None,
        help_text="The shops contributing to this audience",
    )

    # Remote platform data tracking
    approximate_size = PositiveBigIntegerField(
        default=0,
        help_text="The approximate number of people in this audience as described by the platform",
    )

    # Custom Manager
    objects = Manager()
    stale_objects = StaleAudienceManager()

    def _set_break_date(self):
        if self.rolling_days is not None:
            self.break_date = timezone.now() - timedelta(days=self.rolling_days)

        if self.break_date is None:
            self.break_date = timezone.now() - timedelta(days=30)

    def save(self, *args, **kwargs) -> None:
        creating = self.pk is None
        super().save(*args, **kwargs)
        if creating and os.environ.get("ENV").lower() == "production":
            from shadowfax.truestock.tasks import synchronize_custom_audience

            synchronize_custom_audience.delay(self.pk, self.__class__.__name__)

    @abc.abstractmethod
    def update_shop_ids(self) -> bool:
        """
        Ensures that self.shop_ids is up-to-date given an instance's creation parameters.
        Returns TRUE if an update was performed, otherwise FALSE.
        """
        pass

    @abc.abstractmethod
    def synchronize(self) -> None:
        """
        Ensure the remote audience & local audience data are all in sync with one another.
        """
        pass

    @abc.abstractmethod
    def _kwargs(self, **extra_kwargs):
        """
        Return a dict appropriate for passing into an audience client's create_audience method.
        Provide extra keyword args to augment or override this dict.
        """
        pass

    class Meta:
        abstract = True


class CustomFacebookAudience(CustomAudience):
    # FB-Specific audience creation & maintenance parameters
    prefill = BooleanField(default=True)
    source = CharField(max_length=255, blank=True, default="Hyphen Tracking")

    history = HistoricalRecords(table_name="truestock_historicalcustomfacebookaudience")

    def _kwargs(self, **extra_kwargs):
        return {
            "name": self.name,
            "prefill": self.prefill,
            "break_date": self.break_date,
            "brand": self.brand,
            "retailers": self.retailers,
            "vertical": self.vertical,
            "source": self.source,
            "conversions": self.conversions,
        } | {**extra_kwargs}

    def update_shop_ids(self) -> bool:
        from shadowfax.ad_platforms.facebook.client import FacebookMarketingAPIClient

        self._set_break_date()

        client = FacebookMarketingAPIClient()
        kwargs = self._kwargs(dry_run=True)
        _, shop_ids = client.create_audience(**kwargs)

        if not bool(shop_ids):
            return False
        if not bool(self.shop_ids) or set(shop_ids) != set(self.shop_ids):
            self.shop_ids = shop_ids
            return True
        return False

    def archive(self):
        self.archived = True
        self.save()

    def synchronize(self) -> None:
        from shadowfax.ad_platforms.facebook.client import FacebookMarketingAPIClient

        client = FacebookMarketingAPIClient()

        self.last_sync = timezone.now()

        try:
            if not self.update_shop_ids() and self.last_sync_successful:
                # Shop IDs have not changed, and last attempt to sync was fine. Let's update platform audience counts & exit

                fb_audience = FBCA(self.audience_id).api_get(
                    fields=[
                        "approximate_count_lower_bound",
                        "approximate_count_upper_bound",
                    ]
                )
                self.approximate_size = int(
                    (
                        fb_audience["approximate_count_lower_bound"]
                        + fb_audience["approximate_count_upper_bound"]
                    )
                    / 2
                )

                self.save()
                return None

            if self.last_sync_successful is None:
                kwargs = self._kwargs(dry_run=False)
                fb_audience, _ = client.create_audience(**kwargs)
                self.audience_id = fb_audience.get_id()
                self.approximate_size = int(
                    (
                        fb_audience["approximate_count_lower_bound"]
                        + fb_audience["approximate_count_upper_bound"]
                    )
                    / 2
                )

            else:
                client.update_audience_shop_ids(self) # note: this method updates self.approximate_size

        except Exception as e:
            log.warn(f"Failed to synchronize audience {self.audience_id}: {e}")
            self.last_sync_successful = False
        else:
            self.last_sync_successful = True
        finally:
            self.save()

    def __str__(self):
        return f"Facebook Audience {self.audience_id}"

    class JSONAPIMeta:
        resource_name = "custom_facebook_audiences"


class CustomTradeDeskAudience(CustomAudience):
    # External data tracking
    first_party_data_ids = ArrayField(
        PositiveBigIntegerField(), null=True, default=None
    )
    data_group_id = CharField(max_length=255, null=True, blank=True, default=None)

    history = HistoricalRecords(
        table_name="truestock_historicalcustomtradedeskaudience"
    )

    def _kwargs(self, **extra_kwargs):
        return {
            "name": self.name,
            "break_date": self.break_date,
            "brand": self.brand,
            "retailers": self.retailers,
            "vertical": self.vertical,
            "conversions": self.conversions,
        } | {**extra_kwargs}

    def _get_client(self, **kwargs):
        from shadowfax.ad_platforms.the_trade_desk.client import TtdClient

        return TtdClient(**kwargs)

    def update_shop_ids(self):
        self._set_break_date()
        client = self._get_client(auth_token="dry_run")
        kwargs = self._kwargs(dry_run=True)
        _, shop_ids, _, _, _ = client.create_audience(**kwargs)
        if not bool(shop_ids):
            return False
        if not bool(self.shop_ids) or set(shop_ids) != set(self.shop_ids):
            self.shop_ids = shop_ids
            return True
        return False

    def synchronize(self) -> None:
        client = self._get_client()

        self.last_sync = timezone.now()
        if (
            not self.update_shop_ids()
            and self.last_sync_successful
            and bool(self.audience_id)
        ):
            try:
                self.approximate_size = client.get_audience(self.audience_id).json()[
                    "UniqueUserCount"
                ]
            except AttributeError:
                # TODO: parse failure to retrieve & update count here as an indication that this audience should be archived
                pass
            self.save()
            return None

        if self.last_sync_successful is None or not bool(self.audience_id):
            kwargs = self._kwargs(dry_run=False)
            (
                audience_id,
                _,
                first_party_data_ids,
                data_group_id,
                user_count,
            ) = client.create_audience(**kwargs)
            self.audience_id = audience_id
            self.first_party_data_ids = first_party_data_ids
            self.data_group_id = data_group_id
            self.approximate_size = user_count

        else:
            if (
                fpd_ids := client.update_data_group(self)
            ) is not None:  # note: calling this updates self.approximate_size
                self.first_party_data_ids = fpd_ids
            else:
                self.last_sync_successful = False
                self.save()
                return None

        self.last_sync_successful = True
        self.save()
        return None

    def __str__(self):
        return f"TradeDesk Audience {self.audience_id}"

    class JSONAPIMeta:
        resource_name = "custom_trade_desk_audiences"
