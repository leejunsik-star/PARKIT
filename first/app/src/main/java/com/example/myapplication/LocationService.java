package com.example.myapplication;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.location.Location;
import android.os.Build;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;

public class LocationService extends Service {

    private static final String TAG = "LocationService";
    private static final String CHANNEL_ID = "location_service_channel_high";
    private static final int NOTIFICATION_ID = 101;

    private FusedLocationProviderClient fusedLocationClient;
    private LocationCallback locationCallback;

    private double destLat;
    private double destLng;
    private String parkingId;
    private String parkingName;
    private boolean isArrivedTriggered = false; // 도착 트리거 중복 실행 방지 플래그

    @Override
    public void onCreate() {
        super.onCreate();
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
        createNotificationChannel();
    }

    @SuppressLint("ForegroundServiceType")
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            stopSelf();
            return START_NOT_STICKY;
        }

        String action = intent.getAction();
        if ("STOP_SERVICE".equals(action)) {
            Log.d(TAG, "안내 중단 액션 수신: 서비스 종료");
            stopSelf();
            return START_NOT_STICKY;
        }

        destLat = intent.getDoubleExtra("dest_lat", 0.0);
        destLng = intent.getDoubleExtra("dest_lng", 0.0);
        parkingId = intent.getStringExtra("parking_id");
        parkingName = intent.getStringExtra("parking_name");

        if (destLat == 0.0 || destLng == 0.0 || parkingId == null) {
            Log.e(TAG, "올바르지 않은 좌표 및 주차장 ID: 서비스 종료");
            stopSelf();
            return START_NOT_STICKY;
        }

        // 포그라운드 서비스 알림 활성화
        Notification notification = buildNotification("길안내 시작: " + parkingName);
        startForeground(NOTIFICATION_ID, notification);

        startLocationUpdates();

        return START_REDELIVER_INTENT;
    }

    private void startLocationUpdates() {
        LocationRequest locationRequest = new LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 4000)
                .setMinUpdateIntervalMillis(2000)
                .build();

        locationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult locationResult) {
                if (locationResult == null || isArrivedTriggered) return;
                for (Location location : locationResult.getLocations()) {
                    float[] results = new float[1];
                    Location.distanceBetween(location.getLatitude(), location.getLongitude(), destLat, destLng, results);
                    float distance = results[0];

                    Log.d(TAG, "현재 남은 거리: " + distance + "m");

                    if (distance <= 100f) {
                        isArrivedTriggered = true;
                        Log.d(TAG, "목적지 반경 100m 이내 진입! 앱 복귀 및 상세보기 실행");
                        
                        // 즉각적인 백그라운드 GPS 수신 차단 (중복 방지 핵심)
                        if (fusedLocationClient != null) {
                            fusedLocationClient.removeLocationUpdates(this);
                        }
                        
                        triggerArrivedAction();
                        break;
                    } else {
                        updateNotification(String.format(java.util.Locale.getDefault(), "목적지까지 약 %.0f m 남음", distance));
                    }
                }
            }
        };

        try {
            fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper());
        } catch (SecurityException e) {
            Log.e(TAG, "위치 권한 없음: " + e.getMessage());
            stopSelf();
        }
    }

    private void triggerArrivedAction() {
        // 목적지 도착 진동 및 사운드 울림과 함께 상세보기 화면 전면 배치
        Intent intent = new Intent(this, ParkingDetailActivity.class);
        intent.putExtra("parking_id", parkingId);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);

        // 안드로이드 10 이상 백그라운드 액티비티 차단 우회를 위한 Full Screen PendingIntent 설정
        PendingIntent fullScreenPendingIntent = PendingIntent.getActivity(
                this, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("목적지 도착 완료")
                .setContentText(parkingName + " 근처에 도착했습니다. 주차 도면을 불러옵니다.")
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_CALL)
                .setDefaults(Notification.DEFAULT_ALL)
                .setFullScreenIntent(fullScreenPendingIntent, true) // 즉각 화면 노출 우회
                .setAutoCancel(true);

        if (nm != null) {
            nm.notify(NOTIFICATION_ID + 1, builder.build());
        }

        // 즉시 액티비티도 추가 시작 시도
        try {
            startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
        }

        stopSelf();
    }

    private Notification buildNotification(String text) {
        Intent stopIntent = new Intent(this, LocationService.class);
        stopIntent.setAction("STOP_SERVICE");
        PendingIntent stopPendingIntent = PendingIntent.getService(
                this, 0, stopIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Intent openIntent = new Intent(this, MainActivity.class);
        PendingIntent openPendingIntent = PendingIntent.getActivity(
                this, 0, openIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("주차장 목적지 접근 감지 중")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setContentIntent(openPendingIntent)
                .addAction(android.R.drawable.ic_menu_close_clear_cancel, "안내 중단", stopPendingIntent)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setOngoing(true)
                .build();
    }

    private void updateNotification(String text) {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) {
            nm.notify(NOTIFICATION_ID, buildNotification(text));
        }
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "주차장 목적지 접근 안내",
                    NotificationManager.IMPORTANCE_HIGH
            );
            channel.setDescription("목적지 100m 이내 접근 시 자동으로 앱을 복귀시키고 도면을 활성화하는 서비스 채널입니다.");
            channel.enableLights(true);
            channel.enableVibration(true);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (fusedLocationClient != null && locationCallback != null) {
            fusedLocationClient.removeLocationUpdates(locationCallback);
        }
        Log.d(TAG, "LocationService onDestroy");
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
