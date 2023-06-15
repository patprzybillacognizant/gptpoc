import datetime
import json
import time
from collections import defaultdict
from typing import Any, Dict, Type
import requests









from django.conf import settings
from django.db import IntegrityError
from django.db.models import Model
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.campaign import Campaign
from facebook_business.adobjects.customaudience import CustomAudience
from facebook_business.api import Cursor, FacebookAdsApi
from requests.exceptions import HTTPError

from shadowfax.ad_platforms.audience_builder import ShopFinderMixin
from shadowfax.ad_platforms.facebook.transforms import (
    FacebookAdSetTransform,
    FacebookCampaignTransform,
    _FacebookTransform,
)
from shadowfax.ad_platforms.platform import Platform
from shadowfax.hyphen.up.utils import get_in, recursive_flatten
from shadowfax.resource_locking_client.client import Resource, ResourceLockingClient
from shadowfax.truestock.models import (
    CustomFacebookAudience,
    FacebookCityGeolocationEntity,
    FacebookMarketingAdSet,
    FacebookMarketingCampaign,
    FacebookPostalCodeGeolocationEntity,
    OptimizerType,
)
from shadowfax.utils.decorators import cacheable
from shadowfax.utils.redshift import query_for_fb_ad_sets, query_for_fb_campaigns

FACEBOOK_GRAPH_ROOT = "https://graph.facebook.com"

FACEBOOK_OAUTH_URL = f"{FACEBOOK_GRAPH_ROOT}/oauth/access_token"


class ClusterException(Exception):
    pass


class FacebookMarketingAPIClient(Platform, ShopFinderMixin):
    API_VERSION = "14.0"
    AD_ACCOUNT_ID = settings.FB_MARKETING_AD_ACCOUNT_ID

    def __init__(
        self, app_id=None, app_secret=None, ad_account_id=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.rlclient = ResourceLockingClient()
        self.resource = Resource.TRADE_DESK

        self.pre_existing_zip_targets = None

        self.adsets = None
        self.APP_ID = app_id or settings.FB_HYPHEN_APP_ID
        self.APP_SECRET = app_secret or settings.FB_HYPHEN_APP_SECRET
        self.AD_ACCOUNT_ID = ad_account_id or settings.FB_MARKETING_AD_ACCOUNT_ID
        # self.ACCESS_TOKEN = kwargs.get("auth_token") or self.get_access_token()
        self.ACCESS_TOKEN = kwargs.get("auth_token") or settings.FB_LONG_LIVED_TOKEN

        with self.rlclient.managed_lock(self.resource):
            self._api = FacebookAdsApi.init(
                app_id=self.APP_ID,
                app_secret=self.APP_SECRET,
                access_token=self.ACCESS_TOKEN,
            )
            self.ad_account = AdAccount(f"act_{self.AD_ACCOUNT_ID}")

    def _cursor_to_list(self, cursor: Cursor, throttle: bool = True) -> list[Any]:
        """
        Convert a cursor to a python list by grabbing ALL objects on ALL pages.
        Beware, might be chonky.
        """
        with self.rlclient.managed_lock(self.resource):
            if not throttle:
                return [item for item in cursor]
            retval = list()
            for item in cursor:
                retval.append(item)
                time.sleep(
                    0.25
                )  # TODO: kill me, find a better way to handle rate limiting here
            return retval

    def _sync(self, object_type: Type[Campaign] | Type[AdSet]):
        sf_model: Type[Model]
        tf_class: Type[_FacebookTransform]

        if object_type == Campaign:
            self._log("Preparing to sync campaigns...")
            sf_model = FacebookMarketingCampaign
            tf_class = FacebookCampaignTransform
            objects = query_for_fb_campaigns()
            filter_key = "campaign_id__in"
            sf_id_attr = "campaign_id"
        else:
            self._log("Preparing to sync adsets...")
            sf_model = FacebookMarketingAdSet
            tf_class = FacebookAdSetTransform
            objects = query_for_fb_ad_sets()
            filter_key = "adset_id__in"
            sf_id_attr = "adset_id"

        if not objects:
            return False

        object_external_id_list = [o.get("id") for o in objects]
        _filter = {filter_key: object_external_id_list}

        # First, get all objects to update, using the Blackbird Redshift response as our source of truth
        objects_to_update = sf_model.objects.filter(**_filter)
        object_update_hits = [getattr(x, sf_id_attr) for x in objects_to_update]

        # Delete any objects that we're tracking which are absent from our response
        delete_result = sf_model.objects.exclude(**_filter).delete()
        self._log_debug(f"Obsolete objects deleted ({delete_result=}) ")

        # Figure out which objects from the API response need to be created, as opposed to updated
        objects_to_create = [
            obj for obj in objects if obj.get("id") not in object_update_hits
        ]

        # Create new records
        transformer = tf_class()
        new_objects = [transformer.transform_to_model(c) for c in objects_to_create]
        for _no in new_objects:
            # handle location clusters
            if bool(clusters := getattr(_no, "location_cluster_ids", None)):
                for _id in clusters:
                    try:
                        _no.zips.extend(self.get_location_cluster_zips(_id["key"]))  # type: ignore
                    except ClusterException:
                        continue
            _no.save()
        self._log_debug(f"New objects created: {len(new_objects)}")

        # Finally, handle updates to existing records
        _map = {x.get("id"): x for x in objects}
        for update_obj in objects_to_update:
            raw_data = _map[getattr(update_obj, sf_id_attr)]
            transformed_data = transformer.transform(raw_data)
            # handle location clusters
            if bool(clusters := transformed_data.get("location_cluster_ids", None)):
                for _id in clusters:
                    try:
                        transformed_data["zips"].extend(
                            self.get_location_cluster_zips(_id["key"])
                        )
                    except ClusterException:
                        continue

            # if zips is empty here, pop it from the list to avoid overwriting due to bad sync data
            if transformed_data.get("zips") == list():
                del transformed_data["zips"]

            for k, v in transformed_data.items():
                setattr(update_obj, k, v)
            update_obj.save()
        self._log_debug(f"Existing objects updated: {len(objects_to_update)}")

        return True

    def get_location_cluster_zips(self, cluster_id: str):
        url = f"{FACEBOOK_GRAPH_ROOT}/v{self.API_VERSION}/{cluster_id}?access_token={self.ACCESS_TOKEN}&fields=['locations']"
        with self.rlclient.managed_lock(self.resource):
            response = requests.get(url).json()

        # TODO: validation
        if "error" in response:
            raise ClusterException(response["error"]["message"])

        return [
            {"key": x, "name": x.replace("US:", ""), "country": "US"}
            for x in response.get("locations", list())
        ]

    def sync_campaigns_and_adsets(self) -> tuple[bool, bool]:
        """
        Ensure parity between local database and Blackbird Redshift db, using the latter as the source of truth.

        - Create any campaigns / adsets that are not present in our DB, but are present in FB
        - Update any campaigns / adsets that are present in our DB and are present in FB
        - Remove any campaigns / adsets that are present in our DB and are not present in FB

        """
        return self._sync(Campaign), self._sync(AdSet)

    def get_access_token(self, grant_type="client_credentials"):
        self._log_debug("Authenticating to acquire access token")

        oauth_params = {
            "client_id": self.APP_ID,
            "client_secret": self.APP_SECRET,
            "grant_type": grant_type,
        }
        with self.rlclient.managed_lock(self.resource):
            response = requests.get(FACEBOOK_OAUTH_URL, params=oauth_params)

        try:
            response.raise_for_status()
            access_token = response.json().get("access_token")
        except (HTTPError, json.decoder.JSONDecodeError) as e:
            status_code = response.status_code
            content = response.content
            self._log_error(
                f"Failed to get Facebook marketing API access token. Response: {status_code} -- {content}"
            )
            raise e

        return access_token

    def targeting_search(
        self,
        search_type="adgeolocation",
        location_types=None,
        search_term="",
        limit=100000,
    ):
        """
        Finds predefined values that can be used to setup an audience for targeting
        """

        url = f"{FACEBOOK_GRAPH_ROOT}/v{self.API_VERSION}/search"
        params = {
            "type": search_type,
            "q": search_term,
            "access_token": self.ACCESS_TOKEN,
            "country_code": "US",
            "limit": limit,
        }

        if location_types is not None and type(location_types) == list:
            params["location_types"] = location_types
        with self.rlclient.managed_lock(self.resource):
            data = requests.get(url, params).json()

        return data

    def get_fb_audiences(self, **kwargs):
        with self.rlclient.managed_lock(self.resource):
            custom_audiences_cursor = self.ad_account.get_custom_audiences(**kwargs)
        return self._cursor_to_list(custom_audiences_cursor)

    @cacheable("fb_ads_pixels", cache_ttl=60 * 60 * 24 * 7)
    def get_fb_pixels(self):
        with self.rlclient.managed_lock(self.resource):
            ads_pixels_cursor = self.ad_account.get_ads_pixels(fields=["name"])
        pixel_list = self._cursor_to_list(ads_pixels_cursor)
        return [
            {"id": px.get("id"), "type": "pixel", "name": px.get("name")}
            for px in pixel_list
        ]

    def create_audience(
        self,
        name: str,
        prefill: bool = True,
        break_date: datetime.datetime = None,
        brand: str = None,
        retailers: list[str] = None,
        vertical: str = None,
        source: str = None,
        conversions: bool = False,
        shop_ids: list[str] = None,
        dry_run: bool = False,
    ) -> tuple[CustomAudience | None, list[str] | None]:
        """
        Create a FaceBook audience from the given criteria. Note that a maximum of 99 shops may be used as filters
        at one time.

        Args:
            name: The name of the audience to create
            prefill: Should this audience include data from before creation (True) or only from the date of creation?
            break_date: Restrict results to shops created after this date. Defaults to 30 days in the past.
            brand: Restrict results to shops that include a product with a given brand name
            retailers: Restrict results to shops that include products offered by given retailers
            vertical: Restrict results to a given ProductCategory
            source: The FB pixel name from which data should be derived (defaults to "Hyphen Tracking")
            conversions: Build audience from AddToCart (True) or PageView (False) events? Defaults to False
            shop_ids: Use a specific list of shop_ids instead of generating a query; incompatible with other filters
            dry_run: return shop_ids only

        Returns:
            the ID of the custom audience created in FB, or the error returned; along with the shop IDs used to create
            the URL filters for said audience
        """
        if source is None:
            source = "Hyphen Tracking"

        event_source = next(
            (pixel for pixel in self.get_fb_pixels() if pixel["name"] == source), None
        )
        if event_source is None:
            return None, None
        del event_source["name"]

        if shop_ids is None:
            shop_ids = self.get_shops_for_audience(
                break_date, brand, retailers, vertical
            )
        if dry_run or len(shop_ids) == 0:
            return None, shop_ids

        rule = self._construct_rule(
            shop_ids=shop_ids, event_source=event_source, conversions=conversions
        )
        with self.rlclient.managed_lock(self.resource):
            response = self.ad_account.create_custom_audience(
                fields=[
                    "approximate_count_lower_bound",
                    "approximate_count_upper_bound",
                ],
                params={
                    "name": name,
                    "prefill": prefill,
                    "rule": rule,
                },
            )
        return response, shop_ids

    @staticmethod
    def _construct_rule(
        shop_ids: list[str], event_source: dict, conversions: bool
    ) -> str:
        """
        Construct a rule string appropriate for use in custom audience interactions
        """
        # You really, really must be precise with how this whole thing gets constructed. Anything that doesn't match
        # their (half-documented) schema will break the audience builder UI. Worse, it'll appear to create a valid
        # audience -- but that audience will never be populated with data.
        filters = [
            {"field": "url", "operator": "i_contains", "value": _id} for _id in shop_ids
        ]

        rules = [
            {
                "event_sources": [event_source],
                "retention_seconds": 60 * 60 * 24 * 180,
                "filter": {
                    "operator": "and",
                    "filters": [
                        {
                            "field": "event",
                            "operator": "eq",
                            "value": "AddToCart" if conversions else "PageView",
                        },
                        {
                            "operator": "or",
                            "filters": filters[:99],
                        },
                    ],
                },
            }
        ]
        return json.dumps({"inclusions": {"operator": "or", "rules": rules}})

    def update_audience_shop_ids(self, local_audience: CustomFacebookAudience) -> bool:
        event_source = next(
            (
                pixel
                for pixel in self.get_fb_pixels()
                if pixel["name"] == local_audience.source
            ),
            None,
        )
        if event_source is None:
            return False
        del event_source["name"]

        rule = self._construct_rule(
            shop_ids=local_audience.shop_ids,
            event_source=event_source,
            conversions=local_audience.conversions,
        )

        fb_audience = CustomAudience(local_audience.audience_id)
        fb_audience[CustomAudience.Field.rule] = rule
        with self.rlclient.managed_lock(self.resource):
            try:
                response = fb_audience.api_update(
                    fields=[
                        "approximate_count_lower_bound",
                        "approximate_count_upper_bound",
                    ]
                )
                local_audience.approximate_size = int(
                    (
                        response["approximate_count_lower_bound"]
                        + response["approximate_count_upper_bound"]
                    )
                    / 2
                )
            except:  # pokemon
                return False
        return True

    def revert_audience_targeting_optimizations(self, adgroup):
        self.de_optimize_target_audience_for_adset(adgroup)

    def validate_data_on_optimizer(self):
        super().validate_data_on_optimizer()

        self.adsets = self.optimizer.facebook_adsets.all()

        if self.adsets.count() == 0:
            return (
                False,
                f"Optimizer '{self.optimizer.name}' does not have any adsets to optimize",
            )

        return True, ""

    def targeted_postal_codes(self):
        adsets = self.optimizer.facebook_adsets.all()
        pre_existing_zip_targets = recursive_flatten([adset.zips for adset in adsets])

        return list(
            set(
                z if not hasattr(z, "get") else z.get("name")
                for z in pre_existing_zip_targets
            )
        )

    def obtain_required_inputs(self):
        if self.pre_existing_zip_targets is None:
            self.pre_existing_zip_targets = self.targeted_postal_codes()
        return self.pre_existing_zip_targets

    def construct_target_audience_with_truestock_results(
        self,
        bid_data_by_postal,
    ):
        """
        TODO: IF WE'RE HERE, BID_DATA IS ACTUALLY BY FB CITY CODE -- NOT POSTAL. FIX? RENAME? DRINK?
        """
        targeting = {"user_groups": {"home_location": {"cities": {}}}}
        fb_city_ids = list(bid_data_by_postal.keys())
        fb_city_db_records = FacebookCityGeolocationEntity.objects.filter(
            city_id__in=fb_city_ids
        )

        for db_record in fb_city_db_records:
            city_id = db_record.city_id

            bid_adjustment = bid_data_by_postal[city_id].bid_adjustment

            if bid_adjustment != 1.0:
                city_targets = targeting["user_groups"]["home_location"]["cities"]
                if city_targets.get(city_id) is None and "-US" not in city_id:
                    # Meta bid multipliers must be 0.09 >= x >= 1.0
                    if bid_adjustment > 1.0:
                        city_targets[city_id] = 1.0
                    elif bid_adjustment < 0.09:
                        city_targets[city_id] = 0.09
                    else:
                        city_targets[city_id] = bid_adjustment

        # TODO: WE'RE RETURNING CITY_IDS HERE NOW, WHAT DOES THAT BREAK?
        return targeting, fb_city_ids

    def run_inputs_through_truestock(self, postal_codes):
        product_ids = self.optimizer.get_all_product_ids()
        return self.optimizer.populate_bid_data_by_postal_code(
            product_ids, postal_codes
        )

    def de_optimize_target_audience_for_adset(self, adset_id):
        self._log_debug(f"De-optimizing bid adjustments for adset: {adset_id}")

        adset = AdSet(fbid=adset_id, parent_id=f"act_{self.AD_ACCOUNT_ID}")
        with self.rlclient.managed_lock(self.resource):
            adset.update(
                {
                    "bid_adjustments": {},
                }
            )

    def optimize_bid_adjustments_for_adset(self, adset):
        adset_id = adset.adset_id
        self._log(f"API CALL: Optimizing bid adjustments for adset: {adset_id}")

        AdSet(adset_id).update(
            fields=[], params={"bid_adjustments": self.target_audience_template.target}
        )

        return adset_id

    def _get_cities_targeted(self):
        zips = self.pre_existing_zip_targets
        cities = FacebookPostalCodeGeolocationEntity.objects.filter(
            name__in=zips
        ).distinct("primary_city")
        return cities.count()

    def _get_cities_adjusted(self):
        adjusted = get_in(
            self.target_audience_template.target,
            ["user_groups", "home_location", "cities"],
            {},
        )
        return len(adjusted.keys())

    def aggregate_bid_adjustment_count_from_target_cities(self) -> Dict[str, int]:
        bid_adjustment_counts = defaultdict(int)  # type: Dict[str, int]

        # TODO - Since adjustments for social are city scoped, aggregates can only be by city. However,
        # "number of zips targeted" and "number of zips adjusted" are always by zip for social AND display. This is
        # possible because we can get the sum of all zips within the cities targeted. We could technically also do
        # the same for social by assigning the same adjustment to all zips underneath each city, and then aggregating,
        # but we should probably have a conversation about that, as well as adding tooltips for clarity in the Lucy UI
        cities = get_in(
            self.target_audience_template.target,
            ["user_groups", "home_location", "cities"],
            {},
        )
        for city in cities.keys():
            adjustment = str(cities[city])
            bid_adjustment_counts[adjustment] += 1

        return bid_adjustment_counts

    @staticmethod
    def translate_facebook_cities_to_zips(
        filter_postal_codes: list[str] | None, targeted_cities: list[str]
    ) -> list[str]:
        """
        Given a list of targeted FB city geolocations:
            - find all postal codes associated with these cities
            - return all postal codes in this list that are also in the targeted list
        """
        fb_zip_db_records = list(
            postal_code.replace("US:", "")
            for postal_code in FacebookPostalCodeGeolocationEntity.objects.filter(
                primary_city__city_id__in=targeted_cities
            ).values_list("postal_code", flat=True)
        )
        if filter_postal_codes is not None:
            return [z for z in fb_zip_db_records if z in filter_postal_codes]
        return [z for z in fb_zip_db_records]

    def update_target_audience_template(self, bid_data_by_postal_code) -> None:
        """
        Update the target audience template with aggregate counts of bid adjustments
        """
        # get fbk city ids
        (
            self.target_audience_template.target,
            targeted_cities,
        ) = self.construct_target_audience_with_truestock_results(
            bid_data_by_postal_code
        )

        # tabulate bid adjustment
        self.target_audience_template.oos_aggregates = (
            self.optimizer.aggregate_oos_counts(bid_data_by_postal_code)
        )
        self.target_audience_template.oos_zips_by_product_id = (
            self.optimizer.convert_bid_data_by_postal_code_to_oos_zips_by_product_id(
                bid_data_by_postal_code
            )
        )
        bid_adjustment_counts = self.aggregate_bid_adjustment_count_from_target_cities()
        # ⚠️ See To-do on aggregate_bid_adjustment_count_from_target_cities method
        self.target_audience_template.bid_adjustment_aggregates = bid_adjustment_counts
        # ⚠️ Num Targeted should ALWAYS be total zips in campaign targeting
        targeted_postal_codes = self.targeted_postal_codes()
        self.target_audience_template.num_targeted = len(targeted_postal_codes)
        # ⚠️ Num Adjusted should ALWAYS be zips, even though social optimizers adjust cities
        adjusted_postal_codes = self.translate_facebook_cities_to_zips(
            targeted_postal_codes, targeted_cities
        )
        self.target_audience_template.num_adjusted = len(adjusted_postal_codes)

    def update_target_audience_on_platform(self, task_id: str) -> None:
        """
        Save, Post, and Notify
        - Update the target audience on the platform (FBK)
        - Post update on slack
        - Save the processed target audience template
        """

        # update adset on fbk
        processed_adsets = {
            self.optimize_bid_adjustments_for_adset(adset)
            for adset in self.optimizer.facebook_adsets.all()
        }

        # update target audience template
        self.target_audience_template.associated_adset_ids = processed_adsets
        self.target_audience_template.save()

        # ping slack
        args = {
            "bid_adjustment_counts": self.target_audience_template.bid_adjustment_aggregates,
            "targeted": self.target_audience_template.num_targeted,
            "adjusted": self.target_audience_template.num_adjusted,
        }
        self.optimizer.send_slack_notification(
            "successful_optimization", args, task_id=task_id
        )

    @staticmethod
    def _destruct(dictionary: dict, keys: list[str]) -> tuple:
        values = [dictionary.get(k, "") for k in keys]

        return tuple(values)

    def upsert_city_and_postal_code_entities(self, zip_data, has_primary_city: bool):
        c_id, c_name, r_id, r_name, p_code, p_name = self._destruct(
            zip_data,
            ["primary_city_id", "primary_city", "region_id", "region", "key", "name"],
        )

        if not has_primary_city:
            c_id = f"{c_id}-{p_code}"
            c_name = "no-primary-city"

        city, _ = FacebookCityGeolocationEntity.objects.get_or_create(
            city_id=c_id,
            name=c_name,
            region=r_name,
            region_id=r_id,
            country_code="US",
        )

        FacebookPostalCodeGeolocationEntity.objects.get_or_create(
            postal_code=p_code, name=p_name, primary_city=city
        )

    def get_geolocation_data_from_facebook(self, url, params, request_session):
        self._log("Getting geolocation data from Facebook graph API")
        with self.rlclient.managed_lock(self.resource):
            response = request_session.get(url, params=params).json()
        data = response.get("data", [])

        for z in data:
            try:
                self.upsert_city_and_postal_code_entities(z, True)
                self._log("Successfully created missing city and postal code record")
            except IntegrityError as e:
                self.upsert_city_and_postal_code_entities(z, False)

    def populate_city_and_postal_codes_records(self, zipcode):
        SEARCH_URL = "https://graph.facebook.com/v14.0/search"
        ACCESS_TOKEN = self.ACCESS_TOKEN

        params = {
            "type": "adgeolocation",
            "access_token": ACCESS_TOKEN,
            "limit": 10,
            "q": str(zipcode),
            "location_types": ["zip"],
            "country_code": "US",
        }

        s = requests.Session()

        self._log("Populating Facebook graph API city and postal-code data")
        self.get_geolocation_data_from_facebook(SEARCH_URL, params, s)
