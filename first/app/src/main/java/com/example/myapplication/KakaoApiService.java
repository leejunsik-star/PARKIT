package com.example.myapplication;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Header;
import retrofit2.http.Query;

public interface KakaoApiService {
    @GET("v2/local/search/keyword.json")
    Call<KakaoKeywordResponse> searchKeyword(
            @Header("Authorization") String authorization,
            @Query("query") String query
    );
}
