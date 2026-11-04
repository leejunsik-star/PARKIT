package com.example.myapplication;

import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.ArrayList;
import java.util.List;

public class ParkingDetailActivity extends AppCompatActivity {

    private String parkingId;
    private ParkingMapView parkingMapView;
    private TextView tvName, tvCategory, tvFeeType;
    private Spinner spinnerFloor;
    private android.widget.ImageButton ibFavorite;
    private boolean isFavorite = false;
    private ParkingSpot currentParkingSpot = null;
    private android.widget.Button btnDetailNavi;

    // 내 주차 등록 정보 보관
    private String myParkedParkingId = null;
    private String myParkedZoneName = null;
    private String myParkedSpotId = null;
    private UserCarInfo currentUserCarInfo = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_parking_detail);

        parkingId = getIntent().getStringExtra("parking_id");
        if (parkingId == null || parkingId.isEmpty()) {
            Toast.makeText(this, "주차장 정보를 로드할 수 없습니다.", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        tvName = findViewById(R.id.tv_detail_name);
        tvCategory = findViewById(R.id.tv_detail_category);
        tvFeeType = findViewById(R.id.tv_detail_fee_type);
        parkingMapView = findViewById(R.id.parking_map_view);
        spinnerFloor = findViewById(R.id.spinner_floor);
        ibFavorite = findViewById(R.id.ib_detail_favorite);
        btnDetailNavi = findViewById(R.id.btn_detail_navi);

        ibFavorite.setOnClickListener(v -> toggleFavorite());
        btnDetailNavi.setOnClickListener(v -> startNavigation());
        btnDetailNavi.setEnabled(false); // 데이터 수신 완료 전까지 비활성화

        // 주차칸 터치 클릭 이벤트 연동
        parkingMapView.setOnSpotClickListener((zone, spot) -> onSpotClicked(zone, spot));

        // 먼저 사용자 차량 정보 로드 및 바인딩
        loadUserCarInfoAndDetail();
    }

    private void loadUserCarInfoAndDetail() {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user != null) {
            FirebaseFirestore.getInstance()
                    .collection("users")
                    .document(user.getUid())
                    .get()
                    .addOnSuccessListener(doc -> {
                        UserCarInfo carInfo = UserCarInfo.getGuest();
                        if (doc != null && doc.exists()) {
                            carInfo = UserCarInfo.fromDocument(doc);
                            currentUserCarInfo = carInfo;
                            
                            // 즐겨찾기 목록 확인
                            Object favsObj = doc.get("favorites");
                            if (favsObj instanceof List) {
                                List<?> favs = (List<?>) favsObj;
                                isFavorite = favs.contains(parkingId);
                            }

                            // 내 주차 등록 정보 로드
                            myParkedParkingId = doc.getString("parkedParkingId");
                            myParkedZoneName = doc.getString("parkedZoneName");
                            myParkedSpotId = doc.getString("parkedSpotId");
                        } else {
                            myParkedParkingId = null;
                            myParkedZoneName = null;
                            myParkedSpotId = null;
                            currentUserCarInfo = UserCarInfo.getGuest();
                        }
                        updateFavoriteIcon();
                        parkingMapView.setMyParkedInfo(myParkedParkingId, myParkedSpotId);
                        parkingMapView.setCurrentCarInfo(carInfo);
                        loadParkingDetail();
                    })
                    .addOnFailureListener(e -> {
                        updateFavoriteIcon();
                        parkingMapView.setCurrentCarInfo(UserCarInfo.getGuest());
                        loadParkingDetail();
                    });
        } else {
            updateFavoriteIcon();
            parkingMapView.setCurrentCarInfo(UserCarInfo.getGuest());
            loadParkingDetail();
        }
    }

    private void updateFavoriteIcon() {
        ibFavorite.setImageResource(isFavorite 
                ? R.drawable.ic_star_filled 
                : R.drawable.ic_star_border);
    }

    private void toggleFavorite() {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user == null) {
            Toast.makeText(this, "로그인이 필요한 서비스입니다.", Toast.LENGTH_SHORT).show();
            return;
        }

        String uid = user.getUid();
        com.google.firebase.firestore.DocumentReference docRef = FirebaseFirestore.getInstance()
                .collection("users")
                .document(uid);

        if (isFavorite) {
            // 즐겨찾기 해제
            docRef.update("favorites", com.google.firebase.firestore.FieldValue.arrayRemove(parkingId))
                    .addOnSuccessListener(aVoid -> {
                        isFavorite = false;
                        updateFavoriteIcon();
                        Toast.makeText(ParkingDetailActivity.this, "즐겨찾기에서 해제되었습니다.", Toast.LENGTH_SHORT).show();
                    })
                    .addOnFailureListener(e -> Toast.makeText(ParkingDetailActivity.this, "즐겨찾기 해제 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show());
        } else {
            // 즐겨찾기 추가
            docRef.update("favorites", com.google.firebase.firestore.FieldValue.arrayUnion(parkingId))
                    .addOnSuccessListener(aVoid -> {
                        isFavorite = true;
                        updateFavoriteIcon();
                        Toast.makeText(ParkingDetailActivity.this, "즐겨찾기에 추가되었습니다.", Toast.LENGTH_SHORT).show();
                    })
                    .addOnFailureListener(e -> {
                        // 만약 favorites 필드가 아직 생성되지 않은 경우
                        java.util.Map<String, Object> updates = new java.util.HashMap<>();
                        List<String> newFavs = new ArrayList<>();
                        newFavs.add(parkingId);
                        updates.put("favorites", newFavs);
                        docRef.set(updates, com.google.firebase.firestore.SetOptions.merge())
                                .addOnSuccessListener(aVoid2 -> {
                                    isFavorite = true;
                                    updateFavoriteIcon();
                                    Toast.makeText(ParkingDetailActivity.this, "즐겨찾기에 추가되었습니다.", Toast.LENGTH_SHORT).show();
                                })
                                .addOnFailureListener(e2 -> Toast.makeText(ParkingDetailActivity.this, "즐겨찾기 추가 실패: " + e2.getMessage(), Toast.LENGTH_SHORT).show());
                    });
        }
    }

    private void loadParkingDetail() {
        FirebaseFirestore.getInstance()
                .collection("parkingspace")
                .document(parkingId)
                .get()
                .addOnSuccessListener(documentSnapshot -> {
                    if (documentSnapshot == null || !documentSnapshot.exists()) {
                        Toast.makeText(this, "존재하지 않는 주차장입니다.", Toast.LENGTH_SHORT).show();
                        finish();
                        return;
                    }

                    try {
                        ParkingSpot spot = ParkingSpot.fromDocument(documentSnapshot);
                        currentParkingSpot = spot;
                        btnDetailNavi.setEnabled(true);
                        
                        tvName.setText(spot.name != null ? spot.name : "이름 없는 주차장");

                        // 유형 및 요금 한글 맵핑
                        tvCategory.setText(getCategoryKor(spot.category));
                        tvFeeType.setText(getFeeTypeKor(spot.feeType));

                        // 도면 zones 리스트가 존재하면 스피너 바인딩 및 층 전환 연동
                        if (spot.zones != null && !spot.zones.isEmpty()) {
                            setupFloorSpinner(spot);
                        } else {
                            Toast.makeText(this, "등록된 주차 도면이 없습니다.", Toast.LENGTH_SHORT).show();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        Toast.makeText(this, "데이터 파싱 중 오류: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                })
                .addOnFailureListener(e -> {
                    Toast.makeText(this, "상세정보 조회 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    finish();
                });
    }

    private void setupFloorSpinner(ParkingSpot spot) {
        List<String> floorNames = new ArrayList<>();
        for (ParkingSpot.Zone zone : spot.zones) {
            floorNames.add(zone.name != null ? zone.name : "이름 없는 층");
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, floorNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerFloor.setAdapter(adapter);

        spinnerFloor.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position >= 0 && position < spot.zones.size()) {
                    parkingMapView.setZoneData(spot.zones.get(position), parkingId);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
    }

    private String getCategoryKor(String category) {
        if (category == null) return "기타";
        switch (category) {
            case "public": return "공영 주차장";
            case "private": return "민영 주차장";
            case "business": return "상가 주차장";
            default: return "일반 주차장";
        }
    }

    private String getFeeTypeKor(String feeType) {
        if (feeType == null) return "요금 미정";
        switch (feeType) {
            case "paid": return "유료 이용";
            case "free": return "무료 이용";
            case "conditional_free":
            case "conditional free":
                return "조건부 무료";
            default: return "기타 이용";
        }
    }

    private void startNavigation() {
        if (currentParkingSpot == null) return;
        NaviHelper.startNavigation(this, currentParkingSpot.id, currentParkingSpot.name,
                currentParkingSpot.latitude, currentParkingSpot.longitude);
    }

    private void onSpotClicked(ParkingSpot.Zone zone, ParkingSpot.Spot spot) {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user == null) {
            Toast.makeText(this, "로그인이 필요한 서비스입니다.", Toast.LENGTH_SHORT).show();
            return;
        }

        // 1. 본인이 주차한 자리인지 검사 -> 주차 취소 로직
        boolean isMyCarSpot = parkingId.equals(myParkedParkingId) && spot.id.equals(myParkedSpotId);
        if (isMyCarSpot) {
            new androidx.appcompat.app.AlertDialog.Builder(this)
                    .setTitle("주차 취소")
                    .setMessage("주차를 취소하시겠습니까?")
                    .setPositiveButton("예", (dialog, which) -> {
                        registerOrCancelParking(false, zone.id, spot.id, zone.name, spot.type);
                    })
                    .setNegativeButton("아니오", null)
                    .show();
            return;
        }

        // 2. 이미 점유된 자리인 경우 경고
        if (spot.occupied) {
            Toast.makeText(this, "이미 다른 차량이 주차 중인 칸입니다.", Toast.LENGTH_SHORT).show();
            return;
        }

        // 3. 자격 조건 확인 (Eligible)
        boolean isEligible = true;
        if (currentUserCarInfo != null) {
            if (currentUserCarInfo.isGuest) {
                if (!"NORMAL".equals(spot.type)) {
                    isEligible = false;
                }
            } else {
                if ("COMPACT".equals(spot.type) && !currentUserCarInfo.isCompactCar) {
                    isEligible = false;
                } else if ("EV".equals(spot.type) && !currentUserCarInfo.isElectricCar) {
                    isEligible = false;
                } else if ("HANDICAP".equals(spot.type) && !currentUserCarInfo.isDisabled) {
                    isEligible = false;
                } else if ("PATRIOT".equals(spot.type) && !currentUserCarInfo.isNationalMeritRecipient) {
                    isEligible = false;
                } else if ("WOMAN".equals(spot.type) && !currentUserCarInfo.canUseFemaleParking) {
                    isEligible = false;
                }
            }
        }

        if (!isEligible) {
            Toast.makeText(this, "회원님의 차량 정보로 이용할 수 없는 주차칸입니다.", Toast.LENGTH_SHORT).show();
            return;
        }

        // 4. EV 충전 자리 안내 경고 및 일반 주차 확인
        if ("EV".equals(spot.type)) {
            new androidx.appcompat.app.AlertDialog.Builder(this)
                    .setTitle("전기차 전용 주차구역 안내")
                    .setMessage("선택하신 자리는 '전기차 충전 및 전용 주차구역'입니다.\n\n" +
                            "친환경자동차법에 따라 아래 의무 사항을 위반할 경우 10만 원의 과태료가 부과될 수 있으니 각별히 주의하시기 바랍니다.\n\n" +
                            "일반 차량 주차 금지: 내연기관 차량 주차 시 즉시 단속\n" +
                            "급속 충전 시설: 주차 후 1시간(60분) 초과 시 단속\n" +
                            "완속 충전 시설: 주차 후 14시간 초과 시 단속")
                    .setPositiveButton("확인 및 주차", (dialog, which) -> {
                        registerOrCancelParking(true, zone.id, spot.id, zone.name, spot.type);
                    })
                    .setNegativeButton("취소", null)
                    .show();
        } else {
            new androidx.appcompat.app.AlertDialog.Builder(this)
                    .setTitle("주차 등록")
                    .setMessage("선택하신 자리에 주차를 등록하시겠습니까?")
                    .setPositiveButton("확인", (dialog, which) -> {
                        registerOrCancelParking(true, zone.id, spot.id, zone.name, spot.type);
                    })
                    .setNegativeButton("취소", null)
                    .show();
        }
    }

    private void registerOrCancelParking(boolean isRegister, String targetZoneId, String targetSpotId, String targetZoneName, String spotType) {
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user == null) {
            Toast.makeText(this, "로그인이 필요한 서비스입니다.", Toast.LENGTH_SHORT).show();
            return;
        }
        String uid = user.getUid();

        android.app.ProgressDialog progressDialog = new android.app.ProgressDialog(this);
        progressDialog.setMessage("처리 중입니다...");
        progressDialog.setCancelable(false);
        progressDialog.show();

        FirebaseFirestore db = FirebaseFirestore.getInstance();

        db.runTransaction(transaction -> {
            // 1. 사용자 문서 읽기
            com.google.firebase.firestore.DocumentReference userRef = db.collection("users").document(uid);
            com.google.firebase.firestore.DocumentSnapshot userDoc = transaction.get(userRef);

            String prevParkingId = userDoc.getString("parkedParkingId");
            String prevSpotId = userDoc.getString("parkedSpotId");

            // 2. 이전 주차장 문서 읽기
            com.google.firebase.firestore.DocumentReference prevParkingRef = null;
            com.google.firebase.firestore.DocumentSnapshot prevParkingDoc = null;
            if (prevParkingId != null && !prevParkingId.isEmpty()) {
                prevParkingRef = db.collection("parkingspace").document(prevParkingId);
                prevParkingDoc = transaction.get(prevParkingRef);
            }

            // 3. 현재 주차장 문서 읽기
            com.google.firebase.firestore.DocumentReference currentParkingRef = db.collection("parkingspace").document(parkingId);
            com.google.firebase.firestore.DocumentSnapshot currentParkingDoc = null;
            if (prevParkingId != null && prevParkingId.equals(parkingId)) {
                currentParkingDoc = prevParkingDoc;
            } else {
                currentParkingDoc = transaction.get(currentParkingRef);
            }

            // 4. 데이터 갱신 적용
            if (prevParkingId != null && prevParkingId.equals(parkingId)) {
                // 동일 주차장 내부 이동 혹은 동일 주차장 취소
                ParkingSpot spotObj = ParkingSpot.fromDocument(currentParkingDoc);

                // 4-1. 이전 자리 해제
                boolean prevUpdated = false;
                if (prevSpotId != null) {
                    for (ParkingSpot.Zone z : spotObj.zones) {
                        for (ParkingSpot.Spot s : z.spots) {
                            if (prevSpotId.equals(s.id)) {
                                s.occupied = false;
                                prevUpdated = true;
                                break;
                            }
                        }
                        if (prevUpdated) break;
                    }
                }

                // 4-2. 새 자리 등록 (등록 동작인 경우)
                if (isRegister) {
                    boolean nextUpdated = false;
                    for (ParkingSpot.Zone z : spotObj.zones) {
                        for (ParkingSpot.Spot s : z.spots) {
                            if (targetSpotId.equals(s.id)) {
                                if (s.occupied) {
                                    throw new com.google.firebase.firestore.FirebaseFirestoreException("이미 다른 차량이 주차 중인 칸입니다.", com.google.firebase.firestore.FirebaseFirestoreException.Code.ABORTED);
                                }
                                s.occupied = true;
                                nextUpdated = true;
                                break;
                            }
                        }
                        if (nextUpdated) break;
                    }
                    if (!nextUpdated) {
                        throw new com.google.firebase.firestore.FirebaseFirestoreException("선택한 주차칸을 찾을 수 없습니다.", com.google.firebase.firestore.FirebaseFirestoreException.Code.ABORTED);
                    }
                }

                // 주차장 문서 업데이트
                transaction.update(currentParkingRef, "zones", spotObj.getZonesMapList());

            } else {
                // 다른 주차장이거나 이전 주차가 없는 경우
                // 4-1. 이전 주차장 자리 해제 (이전 주차가 있었던 경우)
                if (prevParkingId != null && prevParkingDoc != null && prevParkingDoc.exists() && prevSpotId != null) {
                    ParkingSpot prevSpotObj = ParkingSpot.fromDocument(prevParkingDoc);
                    boolean prevUpdated = false;
                    for (ParkingSpot.Zone z : prevSpotObj.zones) {
                        for (ParkingSpot.Spot s : z.spots) {
                            if (prevSpotId.equals(s.id)) {
                                s.occupied = false;
                                prevUpdated = true;
                                break;
                            }
                        }
                        if (prevUpdated) break;
                    }
                    if (prevUpdated) {
                        transaction.update(prevParkingRef, "zones", prevSpotObj.getZonesMapList());
                    }
                }

                // 4-2. 새 주차장 자리 등록 (등록 동작인 경우)
                if (isRegister) {
                    ParkingSpot currentSpotObj = ParkingSpot.fromDocument(currentParkingDoc);
                    boolean nextUpdated = false;
                    for (ParkingSpot.Zone z : currentSpotObj.zones) {
                        for (ParkingSpot.Spot s : z.spots) {
                            if (targetSpotId.equals(s.id)) {
                                if (s.occupied) {
                                    throw new com.google.firebase.firestore.FirebaseFirestoreException("이미 다른 차량이 주차 중인 칸입니다.", com.google.firebase.firestore.FirebaseFirestoreException.Code.ABORTED);
                                }
                                s.occupied = true;
                                nextUpdated = true;
                                break;
                            }
                        }
                        if (nextUpdated) break;
                    }
                    if (!nextUpdated) {
                        throw new com.google.firebase.firestore.FirebaseFirestoreException("선택한 주차칸을 찾을 수 없습니다.", com.google.firebase.firestore.FirebaseFirestoreException.Code.ABORTED);
                    }

                    transaction.update(currentParkingRef, "zones", currentSpotObj.getZonesMapList());
                }
            }

            // 5. 사용자 문서 주차 정보 업데이트
            if (isRegister) {
                transaction.update(userRef,
                        "parkedParkingId", parkingId,
                        "parkedZoneName", targetZoneName,
                        "parkedSpotId", targetSpotId
                );
            } else {
                transaction.update(userRef,
                        "parkedParkingId", null,
                        "parkedZoneName", null,
                        "parkedSpotId", null
                );
            }

            return null;
        }).addOnSuccessListener(aVoid -> {
            progressDialog.dismiss();
            Toast.makeText(ParkingDetailActivity.this, isRegister ? "주차 등록이 완료되었습니다." : "주차 등록이 취소되었습니다.", Toast.LENGTH_SHORT).show();
            // 로컬 캐시 갱신 및 UI 업데이트
            if (isRegister) {
                myParkedParkingId = parkingId;
                myParkedZoneName = targetZoneName;
                myParkedSpotId = targetSpotId;
            } else {
                myParkedParkingId = null;
                myParkedZoneName = null;
                myParkedSpotId = null;
            }
            parkingMapView.setMyParkedInfo(myParkedParkingId, myParkedSpotId);
            loadParkingDetail(); // DB에서 최신 정보를 다시 로드하여 뷰에 반영
        }).addOnFailureListener(e -> {
            progressDialog.dismiss();
            Toast.makeText(ParkingDetailActivity.this, "처리에 실패했습니다: " + e.getMessage(), Toast.LENGTH_LONG).show();
        });
    }

}
