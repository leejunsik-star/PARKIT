package com.example.myapplication;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.location.Location;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.kakao.vectormap.LatLng;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class FavoriteAdapter extends RecyclerView.Adapter<FavoriteAdapter.ViewHolder> {

    private final List<ParkingSpot> items = new ArrayList<>();
    private UserCarInfo currentCarInfo = UserCarInfo.getGuest();
    private LatLng myLocation = null;
    private final OnItemClickListener clickListener;

    public interface OnItemClickListener {
        void onItemClick(ParkingSpot spot);
    }

    public FavoriteAdapter(OnItemClickListener listener) {
        this.clickListener = listener;
    }

    public void setData(List<ParkingSpot> newItems, UserCarInfo carInfo) {
        this.items.clear();
        if (newItems != null) {
            this.items.addAll(newItems);
        }
        if (carInfo != null) {
            this.currentCarInfo = carInfo;
        }
        notifyDataSetChanged();
    }

    public void updateMyLocation(LatLng loc) {
        this.myLocation = loc;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_favorite_card, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        ParkingSpot spot = items.get(position);
        Context ctx = holder.itemView.getContext();

        holder.tvName.setText(spot.name != null ? spot.name : "주차장");

        // 잔여 대수 계산 및 연산
        int[] calc = ParkingMarkerHelper.calculateRemaining(spot, currentCarInfo);
        int remaining = calc[0];
        int total = calc[1];
        holder.tvSpots.setText(remaining + " / " + total + "대");

        // 요금 배지 디자인 매핑
        if (spot.feeType == null) {
            holder.tvFee.setText("기타");
            holder.tvFee.setBackgroundColor(Color.parseColor("#9E9E9E"));
        } else {
            switch (spot.feeType) {
                case "free":
                    holder.tvFee.setText("무료");
                    holder.tvFee.setBackgroundColor(Color.parseColor("#4CAF50")); // 초록
                    break;
                case "paid":
                    holder.tvFee.setText("유료");
                    holder.tvFee.setBackgroundColor(Color.parseColor("#FF9800")); // 주황
                    break;
                case "conditional_free":
                case "conditional free":
                    holder.tvFee.setText("조건부");
                    holder.tvFee.setBackgroundColor(Color.parseColor("#78909C")); // Slate 회색
                    break;
                default:
                    holder.tvFee.setText("기타");
                    holder.tvFee.setBackgroundColor(Color.parseColor("#9E9E9E"));
                    break;
            }
        }

        // 거리 정보 계산
        if (myLocation != null) {
            float[] results = new float[1];
            Location.distanceBetween(myLocation.getLatitude(), myLocation.getLongitude(), spot.latitude, spot.longitude, results);
            float distanceMeters = results[0];
            if (distanceMeters >= 1000) {
                holder.tvDistance.setText(String.format(Locale.getDefault(), "내 위치에서 %.1f km", distanceMeters / 1000f));
            } else {
                holder.tvDistance.setText(String.format(Locale.getDefault(), "내 위치에서 %d m", (int) distanceMeters));
            }
        } else {
            holder.tvDistance.setText("거리 정보 없음");
        }

        // 전체 카드 클릭 시 상세 페이지 이동
        holder.itemView.setOnClickListener(v -> {
            if (clickListener != null) {
                clickListener.onItemClick(spot);
            }
        });

        // 길찾기 버튼 연동
        holder.btnNavi.setOnClickListener(v -> {
            NaviHelper.startNavigation(ctx, spot.id, spot.name, spot.latitude, spot.longitude);
        });
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView tvName, tvFee, tvSpots, tvDistance;
        Button btnNavi;

        ViewHolder(View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_card_name);
            tvFee = itemView.findViewById(R.id.tv_card_fee);
            tvSpots = itemView.findViewById(R.id.tv_card_spots);
            tvDistance = itemView.findViewById(R.id.tv_card_distance);
            btnNavi = itemView.findViewById(R.id.btn_card_navi);
        }
    }
}
