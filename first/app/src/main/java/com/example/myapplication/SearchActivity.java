package com.example.myapplication;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.view.ViewGroup;
import android.view.View;
import android.view.LayoutInflater;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class SearchActivity extends AppCompatActivity {

    private EditText etSearch;
    private RecyclerView rvSearchResults;
    private SearchAdapter adapter;
    private String restApiKey;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search);
        
        try {
            android.content.pm.ApplicationInfo appInfo = getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
            restApiKey = appInfo.metaData.getString("com.kakao.REST_API_KEY");
        } catch (Exception e) {
            e.printStackTrace();
        }

        etSearch = findViewById(R.id.et_search);
        ImageButton btnBack = findViewById(R.id.btn_back);
        ImageButton btnSearch = findViewById(R.id.btn_search);
        rvSearchResults = findViewById(R.id.rv_search_results);

        rvSearchResults.setLayoutManager(new LinearLayoutManager(this));
        adapter = new SearchAdapter(place -> {
            Intent intent = new Intent();
            intent.putExtra("latitude", Double.parseDouble(place.y));
            intent.putExtra("longitude", Double.parseDouble(place.x));
            setResult(RESULT_OK, intent);
            finish();
        });
        rvSearchResults.setAdapter(adapter);

        btnBack.setOnClickListener(v -> finish());
        
        btnSearch.setOnClickListener(v -> performSearch());

        etSearch.setOnEditorActionListener((v, actionId, event) -> {
            performSearch();
            return true;
        });
    }

    private void performSearch() {
        String query = etSearch.getText().toString().trim();
        if (query.isEmpty()) return;
        if (restApiKey == null || restApiKey.isEmpty()) {
            Toast.makeText(this, "REST API KEY가 설정되지 않았습니다.", Toast.LENGTH_SHORT).show();
            return;
        }

        KakaoApiService api = RetrofitClient.getApiService();
        api.searchKeyword("KakaoAK " + restApiKey, query).enqueue(new Callback<KakaoKeywordResponse>() {
            @Override
            public void onResponse(Call<KakaoKeywordResponse> call, Response<KakaoKeywordResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    adapter.setItems(response.body().documents);
                } else {
                    Toast.makeText(SearchActivity.this, "검색 실패", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<KakaoKeywordResponse> call, Throwable t) {
                Toast.makeText(SearchActivity.this, "네트워크 오류", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // 어댑터 내부 클래스
    interface OnItemClickListener {
        void onItemClick(KakaoKeywordResponse.Place place);
    }

    class SearchAdapter extends RecyclerView.Adapter<SearchAdapter.ViewHolder> {
        private List<KakaoKeywordResponse.Place> items = new ArrayList<>();
        private final OnItemClickListener listener;

        SearchAdapter(OnItemClickListener listener) {
            this.listener = listener;
        }

        void setItems(List<KakaoKeywordResponse.Place> newItems) {
            this.items = newItems;
            notifyDataSetChanged();
        }

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_search_result, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            KakaoKeywordResponse.Place place = items.get(position);
            holder.tvPlaceName.setText(place.placeName);
            holder.tvAddressName.setText(place.roadAddressName != null && !place.roadAddressName.isEmpty() ? place.roadAddressName : place.addressName);
            holder.itemView.setOnClickListener(v -> listener.onItemClick(place));
        }

        @Override
        public int getItemCount() {
            return items.size();
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView tvPlaceName, tvAddressName;

            ViewHolder(@NonNull View itemView) {
                super(itemView);
                tvPlaceName = itemView.findViewById(R.id.tv_place_name);
                tvAddressName = itemView.findViewById(R.id.tv_address_name);
            }
        }
    }
}
