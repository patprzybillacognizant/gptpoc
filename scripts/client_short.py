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
