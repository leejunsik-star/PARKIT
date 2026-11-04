package com.example.myapplication;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.provider.Settings;
import android.widget.Toast;

public class NaviHelper {

    /**
     * 주차장 길찾기를 수행하고, 백그라운드에서 반경 50m 목적지 도달을 감시하는 서비스를 시작합니다.
     */
    public static void startNavigation(Context context, String parkingId, String parkingName, double lat, double lng) {
        if (context == null || parkingId == null || lat == 0.0 || lng == 0.0) return;

        // '다른 앱 위에 표시' (오버레이) 권한 체크
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(context)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + context.getPackageName()));
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(intent);
                Toast.makeText(context, "도착 시 주차 도면 화면 자동 실행을 위해 '다른 앱 위에 표시' 권한을 허용해 주세요.", Toast.LENGTH_LONG).show();
                return; // 권한 획득 유도를 위해 중단
            }
        }

        // 1. 백그라운드 위치 추적 포그라운드 서비스 실행
        Intent serviceIntent = new Intent(context, LocationService.class);
        serviceIntent.putExtra("dest_lat", lat);
        serviceIntent.putExtra("dest_lng", lng);
        serviceIntent.putExtra("parking_id", parkingId);
        serviceIntent.putExtra("parking_name", parkingName);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent);
        } else {
            context.startService(serviceIntent);
        }

        // 2. 외부 맵 앱 (카카오맵 / 구글맵) 길찾기 시작
        try {
            String url = "kakaomap://route?ep=" + lat + "," + lng + "&by=CAR";
            Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
        } catch (Exception e) {
            String url = "https://www.google.com/maps/dir/?api=1&destination=" + lat + "," + lng;
            Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
        }
    }
}
