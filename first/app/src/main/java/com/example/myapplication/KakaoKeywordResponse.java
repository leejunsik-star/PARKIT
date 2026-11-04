package com.example.myapplication;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class KakaoKeywordResponse {
    @SerializedName("documents")
    public List<Place> documents;

    public static class Place {
        @SerializedName("place_name")
        public String placeName;

        @SerializedName("road_address_name")
        public String roadAddressName;

        @SerializedName("address_name")
        public String addressName;

        @SerializedName("x")
        public String x; // longitude

        @SerializedName("y")
        public String y; // latitude
    }
}
