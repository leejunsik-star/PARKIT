package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.Toast;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;

import android.content.Intent;
import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import com.kakao.vectormap.KakaoMap;
import com.kakao.vectormap.KakaoMapReadyCallback;
import com.kakao.vectormap.KakaoMapSdk;
import com.kakao.vectormap.LatLng;
import com.kakao.vectormap.MapLifeCycleCallback;
import com.kakao.vectormap.MapView;
import com.kakao.vectormap.camera.CameraPosition;
import com.kakao.vectormap.camera.CameraUpdateFactory;
import com.kakao.vectormap.label.Label;
import com.kakao.vectormap.label.LabelLayer;
import com.kakao.vectormap.label.LabelOptions;
import com.kakao.vectormap.label.LabelStyle;
import com.kakao.vectormap.label.LabelStyles;

import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.Priority;
import android.location.Location;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.firebase.firestore.WriteBatch;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.CollectionReference;

public class MainActivity extends AppCompatActivity {

    private MapView mapView;
    private KakaoMap kakaoMap;
    private FusedLocationProviderClient fusedLocationClient;
    private LocationCallback locationCallback;
    private LatLng latestLatLng;
    private boolean isTracking = false;
    private DrawerLayout drawerLayout;
    private ImageButton btnProfile;
    private android.view.View zoomWarningView;

    // ── 주차장 데이터 캐시 ──
    private final List<ParkingSpot> allParkingSpots = new ArrayList<>();
    private UserCarInfo currentCarInfo = null;
    // label ID → Label 객체 (visibility 제어용)
    private final Map<String, Label> parkingLabelMap = new HashMap<>();
    
    // ── 즐겨찾기 요약 카드 UI 변수 ──
    private androidx.recyclerview.widget.RecyclerView rvFavorites;
    private FavoriteAdapter favoriteAdapter;
    private final List<String> favoriteIds = new ArrayList<>();
    private android.view.View bottomSheetFavorites;

    private final ActivityResultLauncher<Intent> searchLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    double lat = result.getData().getDoubleExtra("latitude", 0);
                    double lng = result.getData().getDoubleExtra("longitude", 0);
                    if (lat != 0 && lng != 0 && kakaoMap != null) {
                        kakaoMap.moveCamera(CameraUpdateFactory.newCenterPosition(LatLng.from(lat, lng)));
                    }
                }
            });

    private final ActivityResultLauncher<String[]> locationPermissionRequest = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(), result -> {
                Boolean fine = result.getOrDefault(Manifest.permission.ACCESS_FINE_LOCATION, false);
                Boolean coarse = result.getOrDefault(Manifest.permission.ACCESS_COARSE_LOCATION, false);
                if ((fine != null && fine) || (coarse != null && coarse)) {
                    startLocationUpdates(true);
                } else {
                    Toast.makeText(this, "위치 권한이 거부되어 기본 위치를 표시합니다.", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 카카오맵 SDK 초기화
        String appKey = "";
        try {
            android.content.pm.ApplicationInfo appInfo = getPackageManager().getApplicationInfo(getPackageName(),
                    PackageManager.GET_META_DATA);
            appKey = appInfo.metaData.getString("com.kakao.vectormap.APP_KEY");
            Log.d("KakaoMap", "Loaded APP_KEY: " + appKey);
            android.content.pm.PackageInfo packageInfo = getPackageManager().getPackageInfo(getPackageName(),
                    PackageManager.GET_SIGNATURES);
            for (android.content.pm.Signature sig : packageInfo.signatures) {
                java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA");
                md.update(sig.toByteArray());
                Log.d("KakaoMap", "Device KeyHash: " +
                        android.util.Base64.encodeToString(md.digest(), android.util.Base64.NO_WRAP));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        KakaoMapSdk.init(this, appKey);

        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
        locationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult r) {
                if (r == null || kakaoMap == null)
                    return;
                for (Location loc : r.getLocations()) {
                    latestLatLng = LatLng.from(loc.getLatitude(), loc.getLongitude());
                    updateMyLocationMarker(latestLatLng, false);
                }
            }
        };

        mapView = findViewById(R.id.map_view);
        if (mapView != null) {
            mapView.start(new MapLifeCycleCallback() {
                @Override
                public void onMapDestroy() {
                    Log.d("KakaoMap", "onMapDestroy");
                }

                @Override
                public void onMapError(Exception e) {
                    Log.e("KakaoMap", "onMapError: ", e);
                }
            }, new KakaoMapReadyCallback() {
                @Override
                public void onMapReady(KakaoMap map) {
                    Log.d("KakaoMap", "onMapReady");
                    kakaoMap = map;

                    // 나침반 위젯
                    try {
                        if (kakaoMap.getCompass() != null) {
                            kakaoMap.getCompass().show();
                            kakaoMap.getCompass().setPosition(
                                    com.kakao.vectormap.MapGravity.TOP | com.kakao.vectormap.MapGravity.RIGHT,
                                    30, 200);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    checkLocationPermissionAndStartUpdates(true);

                    // ── 카메라 정지 시 뷰포트 마커 갱신 ──
                    kakaoMap.setOnCameraMoveEndListener(
                            (kakaoMap1, cameraPosition, gestureType) -> updateParkingMarkersForViewport());

                    // ── 라벨 클릭 리스너 (상세화면 연동) ──
                    kakaoMap.setOnLabelClickListener((kakaoMap1, layer, label) -> {
                        Object tag = label.getTag();
                        if (tag != null) {
                            String parkingId = tag.toString();
                            Intent intent = new Intent(MainActivity.this, ParkingDetailActivity.class);
                            intent.putExtra("parking_id", parkingId);
                            startActivity(intent);
                        }
                        return true;
                    });

                    // ── 주차장 데이터 최초 1회 로드 (GL 초기화 완료 대기) ──
                    new Handler(Looper.getMainLooper()).postDelayed(
                            () -> loadParkingData(), 1500);
                }
            });
        }

        zoomWarningView = findViewById(R.id.cv_zoom_warning);
        drawerLayout = findViewById(R.id.drawer_layout);
        android.widget.ImageButton btnMenu = findViewById(R.id.btn_menu);
        if (btnMenu != null)
            btnMenu.setOnClickListener(v -> drawerLayout.openDrawer(GravityCompat.START));

        btnProfile = findViewById(R.id.btn_profile);
        if (btnProfile != null)
            btnProfile.setOnClickListener(v -> startActivity(new Intent(this, ProfileActivity.class)));

        android.widget.TextView tvSearchHint = findViewById(R.id.tv_search_hint);
        if (tvSearchHint != null)
            tvSearchHint.setOnClickListener(v -> searchLauncher.launch(new Intent(this, SearchActivity.class)));

        com.google.android.material.floatingactionbutton.FloatingActionButton fab = findViewById(R.id.fab_my_location);
        if (fab != null) {
            fab.setOnClickListener(v -> {
                if (ActivityCompat.checkSelfPermission(this,
                        Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                    fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                            .addOnSuccessListener(this, loc -> {
                                if (loc != null && kakaoMap != null) {
                                    latestLatLng = LatLng.from(loc.getLatitude(), loc.getLongitude());
                                    updateMyLocationMarker(latestLatLng, true);
                                }
                            });
                } else {
                    checkLocationPermissionAndStartUpdates(true);
                }
            });
        }

        // 즐겨찾기 RecyclerView 및 BottomSheet 초기화
        bottomSheetFavorites = findViewById(R.id.bottom_sheet_favorites);
        rvFavorites = findViewById(R.id.rv_favorites);
        if (rvFavorites != null) {
            rvFavorites.setLayoutManager(new androidx.recyclerview.widget.LinearLayoutManager(this,
                    androidx.recyclerview.widget.LinearLayoutManager.VERTICAL, false));
            favoriteAdapter = new FavoriteAdapter(spot -> {
                Intent intent = new Intent(MainActivity.this, ParkingDetailActivity.class);
                intent.putExtra("parking_id", spot.id);
                startActivity(intent);
            });
            rvFavorites.setAdapter(favoriteAdapter);
        }

    }

    @Override
    protected void onResume() {
        super.onResume();
        updateProfileButton();
        checkUserCarInfoAndRefresh();
    }

    private void checkUserCarInfoAndRefresh() {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user != null) {
            FirebaseFirestore.getInstance()
                    .collection("users").document(user.getUid()).get()
                    .addOnSuccessListener(doc -> {
                        UserCarInfo latestInfo = (doc != null && doc.exists())
                                ? UserCarInfo.fromDocument(doc)
                                : UserCarInfo.getGuest();
                        
                        // 즐겨찾기 목록 확인
                        favoriteIds.clear();
                        if (doc != null && doc.exists()) {
                            Object favsObj = doc.get("favorites");
                            if (favsObj instanceof List) {
                                List<?> favs = (List<?>) favsObj;
                                for (Object f : favs) {
                                    if (f instanceof String) {
                                        favoriteIds.add((String) f);
                                    }
                                }
                            }
                        }

                        if (currentCarInfo == null || isCarInfoChanged(currentCarInfo, latestInfo)) {
                            currentCarInfo = latestInfo;
                            refreshParkingMarkers();
                        }
                        
                        updateFavoritesUI();
                    })
                    .addOnFailureListener(e -> {
                    });
        } else {
            favoriteIds.clear();
            UserCarInfo latestInfo = UserCarInfo.getGuest();
            if (currentCarInfo == null || isCarInfoChanged(currentCarInfo, latestInfo)) {
                currentCarInfo = latestInfo;
                refreshParkingMarkers();
            }
            updateFavoritesUI();
        }
    }

    private void updateFavoritesUI() {
        if (rvFavorites == null || favoriteAdapter == null || bottomSheetFavorites == null) return;
        if (favoriteIds.isEmpty() || allParkingSpots.isEmpty()) {
            bottomSheetFavorites.setVisibility(android.view.View.GONE);
            return;
        }
        List<ParkingSpot> favSpots = new ArrayList<>();
        for (ParkingSpot spot : allParkingSpots) {
            if (favoriteIds.contains(spot.id)) {
                favSpots.add(spot);
            }
        }
        if (favSpots.isEmpty()) {
            bottomSheetFavorites.setVisibility(android.view.View.GONE);
        } else {
            favoriteAdapter.setData(favSpots, currentCarInfo);
            if (latestLatLng != null) {
                favoriteAdapter.updateMyLocation(latestLatLng);
            }
            bottomSheetFavorites.setVisibility(android.view.View.VISIBLE);
        }
    }

    private boolean isCarInfoChanged(UserCarInfo oldInfo, UserCarInfo newInfo) {
        if (oldInfo.isGuest != newInfo.isGuest) return true;
        if (oldInfo.isCompactCar != newInfo.isCompactCar) return true;
        if (oldInfo.isElectricCar != newInfo.isElectricCar) return true;
        if (oldInfo.isDisabled != newInfo.isDisabled) return true;
        if (oldInfo.isNationalMeritRecipient != newInfo.isNationalMeritRecipient) return true;
        if (oldInfo.canUseFemaleParking != newInfo.canUseFemaleParking) return true;
        return false;
    }

    private void refreshParkingMarkers() {
        if (kakaoMap == null || allParkingSpots.isEmpty() || currentCarInfo == null) return;
        Log.d("Parking", "차량 정보 변경 감지: 마커 갱신 시작");
        for (ParkingSpot spot : allParkingSpots) {
            Label label = parkingLabelMap.get(spot.id);
            if (label != null) {
                try {
                    Bitmap bmp = ParkingMarkerHelper.createMarkerBitmap(this, spot, currentCarInfo);
                    LabelStyles newStyles = kakaoMap.getLabelManager()
                            .addLabelStyles(LabelStyles.from(LabelStyle.from(bmp).setAnchorPoint(0.5f, 1.0f)));
                    label.changeStyles(newStyles);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void updateProfileButton() {
        if (btnProfile == null)
            return;
        boolean loggedIn = FirebaseAuth.getInstance().getCurrentUser() != null;
        btnProfile.setColorFilter(Color.parseColor(loggedIn ? "#4361EE" : "#808080"));
    }

    // ──────────────────────────────────────────────────────────────────
    // 주차장 데이터 로드 (최초 1회)
    // ──────────────────────────────────────────────────────────────────

    /**
     * 로그인 여부에 따라 carInfo 로드 후 Firestore에서 전체 주차장 데이터를 1회 가져와 캐시합니다.
     */
    private void loadParkingData() {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user != null) {
            FirebaseFirestore.getInstance()
                    .collection("users").document(user.getUid()).get()
                    .addOnSuccessListener(doc -> {
                        currentCarInfo = (doc != null && doc.exists())
                                ? UserCarInfo.fromDocument(doc)
                                : UserCarInfo.getGuest();
                        fetchAllParkingSpots();
                    })
                    .addOnFailureListener(e -> {
                        currentCarInfo = UserCarInfo.getGuest();
                        fetchAllParkingSpots();
                    });
        } else {
            currentCarInfo = UserCarInfo.getGuest();
            fetchAllParkingSpots();
        }
    }

    /**
     * Firestore parkingspace 컬렉션 전체를 1회 로드하여 메모리에 캐시합니다.
     * 이후 뷰포트 필터링은 캐시에서 수행하므로 추가 네트워크 요청이 없습니다.
     */
    private void fetchAllParkingSpots() {
        FirebaseFirestore.getInstance()
                .collection("parkingspace")
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    if (kakaoMap == null)
                        return;
                    allParkingSpots.clear();
                    parkingLabelMap.clear();

                    LabelLayer layer = kakaoMap.getLabelManager().getLayer();
                    if (layer == null)
                        return;

                    for (DocumentSnapshot doc : querySnapshot.getDocuments()) {
                        try {
                            ParkingSpot spot = ParkingSpot.fromDocument(doc);
                            if (spot.latitude == 0 && spot.longitude == 0)
                                continue;

                            // 마커 Bitmap 생성 후 레이어에 추가 (처음엔 숨김)
                            Bitmap bmp = ParkingMarkerHelper.createMarkerBitmap(
                                    MainActivity.this, spot, currentCarInfo);

                            LabelStyles styles = kakaoMap.getLabelManager()
                                    .addLabelStyles(LabelStyles.from(LabelStyle.from(bmp).setAnchorPoint(0.5f, 1.0f)));
                            LabelOptions opts = LabelOptions
                                    .from("parking_" + spot.id, LatLng.from(spot.latitude, spot.longitude))
                                    .setStyles(styles)
                                    .setTag(spot.id);

                            Label label = layer.addLabel(opts);
                            allParkingSpots.add(spot);
                            if (label != null) {
                                parkingLabelMap.put(spot.id, label);
                                // 초기에는 숨김 상태로 시작
                                try {
                                    label.hide();
                                } catch (Exception ignored) {
                                }
                            }
                        } catch (Exception e) {
                            Log.e("Parking", "마커 생성 오류: " + e.getMessage());
                        }
                    }
                    Log.d("Parking", "주차장 " + allParkingSpots.size() + "개 캐시 완료");

                    // 현재 화면에 보이는 마커만 표시
                    updateParkingMarkersForViewport();
                    
                    // 즐겨찾기 카드 UI 업데이트
                    updateFavoritesUI();
                })
                .addOnFailureListener(e -> Log.e("Parking", "parkingspace 로드 실패: " + e.getMessage()));
    }

    // ──────────────────────────────────────────────────────────────────
    // 뷰포트 기반 마커 표시/숨김
    // ──────────────────────────────────────────────────────────────────

    /**
     * 카메라가 멈출 때마다 호출됩니다.
     * 현재 화면 영역에 있는 주차장 마커만 표시하고 나머지는 숨깁니다.
     */
    private void updateParkingMarkersForViewport() {
        if (kakaoMap == null || allParkingSpots.isEmpty() || currentCarInfo == null)
            return;

        CameraPosition cam = kakaoMap.getCameraPosition();
        if (cam == null)
            return;

        double centerLat = cam.getPosition().getLatitude();
        double centerLng = cam.getPosition().getLongitude();
        int zoom = (int) cam.getZoomLevel();

        // ── 줌 레벨이 너무 낮으면 마커를 모두 숨기고 안내 메시지 표시 ──
        int MIN_ZOOM_LEVEL = 14; // 상황에 따라 13~15 사이로 조절 가능
        if (zoom < MIN_ZOOM_LEVEL) {
            if (zoomWarningView != null)
                zoomWarningView.setVisibility(android.view.View.VISIBLE);
            for (ParkingSpot spot : allParkingSpots) {
                Label label = parkingLabelMap.get(spot.id);
                if (label != null) {
                    try {
                        label.hide();
                    } catch (Exception ignored) {
                    }
                }
            }
            return;
        }

        // ── 줌 레벨이 충분하면 안내 메시지 숨기고 뷰포트 내 마커만 표시 ──
        if (zoomWarningView != null)
            zoomWarningView.setVisibility(android.view.View.GONE);

        // 줌 레벨에 따른 화면 범위 계산 (경위도 기준)
        // zoom 15 ≈ ±0.05°, zoom 17 ≈ ±0.013°
        double margin = 180.0 / Math.pow(2, zoom);

        double minLat = centerLat - margin;
        double maxLat = centerLat + margin;
        double minLng = centerLng - margin;
        double maxLng = centerLng + margin;

        int shown = 0;
        for (ParkingSpot spot : allParkingSpots) {
            Label label = parkingLabelMap.get(spot.id);
            if (label == null)
                continue;

            boolean inViewport = spot.latitude >= minLat && spot.latitude <= maxLat
                    && spot.longitude >= minLng && spot.longitude <= maxLng;
            try {
                if (inViewport) {
                    label.show();
                } else {
                    label.hide();
                }
            } catch (Exception e) {
                // hide/show API가 없는 경우 무시
            }
            if (inViewport)
                shown++;
        }
        Log.d("Parking", "화면 내 주차장 " + shown + "/" + allParkingSpots.size() + "개 표시");
    }

    // ──────────────────────────────────────────────────────────────────
    // 내 위치 마커
    // ──────────────────────────────────────────────────────────────────

    private void checkLocationPermissionAndStartUpdates(boolean forceCamera) {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this,
                        Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            startLocationUpdates(forceCamera);
        } else {
            locationPermissionRequest.launch(new String[] {
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
            });
        }
    }

    private void startLocationUpdates(boolean forceCamera) {
        if (ActivityCompat.checkSelfPermission(this,
                Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this,
                        Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED)
            return;

        fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener(this, loc -> {
                    if (loc != null && kakaoMap != null) {
                        latestLatLng = LatLng.from(loc.getLatitude(), loc.getLongitude());
                        updateMyLocationMarker(latestLatLng, forceCamera);
                    }
                });

        if (!isTracking) {
            LocationRequest req = new LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 10000)
                    .setMinUpdateIntervalMillis(10000).build();
            fusedLocationClient.requestLocationUpdates(req, locationCallback, Looper.getMainLooper());
            isTracking = true;
        }
    }

    private void updateMyLocationMarker(LatLng pos, boolean moveCamera) {
        if (kakaoMap == null || pos == null)
            return;

        if (favoriteAdapter != null) {
            favoriteAdapter.updateMyLocation(pos);
        }

        if (moveCamera)
            kakaoMap.moveCamera(CameraUpdateFactory.newCenterPosition(pos));

        LabelLayer layer = kakaoMap.getLabelManager().getLayer();
        if (layer == null)
            return;

        Label label = layer.getLabel("my_location");
        if (label != null) {
            label.moveTo(pos);
        } else {
            android.graphics.Bitmap bmp = android.graphics.Bitmap.createBitmap(80, 80,
                    android.graphics.Bitmap.Config.ARGB_8888);
            android.graphics.Canvas cv = new android.graphics.Canvas(bmp);
            android.graphics.Paint p = new android.graphics.Paint();
            p.setAntiAlias(true);
            p.setColor(android.graphics.Color.WHITE);
            p.setShadowLayer(4f, 0f, 2f, android.graphics.Color.argb(128, 0, 0, 0));
            cv.drawCircle(40, 40, 26, p);
            p.clearShadowLayer();
            p.setColor(android.graphics.Color.RED);
            cv.drawCircle(40, 40, 16, p);

            LabelStyles styles = kakaoMap.getLabelManager()
                    .addLabelStyles(LabelStyles.from(LabelStyle.from(bmp)));
            layer.addLabel(LabelOptions.from("my_location", pos).setStyles(styles));
        }
    }

}