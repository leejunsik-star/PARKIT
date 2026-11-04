package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;

public class ProfileActivity extends AppCompatActivity {

    private FirebaseAuth mAuth;
    private FirebaseFirestore db;
    private LinearLayout layoutLoggedIn;
    private LinearLayout layoutLoggedOut;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_profile);

        mAuth = FirebaseAuth.getInstance();
        db    = FirebaseFirestore.getInstance();

        layoutLoggedIn  = findViewById(R.id.layout_logged_in);
        layoutLoggedOut = findViewById(R.id.layout_logged_out);

        // 공통 닫기 버튼
        MaterialButton btnClose = findViewById(R.id.btn_close_profile);
        btnClose.setOnClickListener(v -> finish());

        // 로그아웃 버튼
        MaterialButton btnLogout = findViewById(R.id.btn_logout);
        btnLogout.setOnClickListener(v -> {
            mAuth.signOut();
            updateUI();
        });

        // 정보 수정 버튼
        MaterialButton btnEditProfile = findViewById(R.id.btn_edit_profile);
        btnEditProfile.setOnClickListener(v ->
                startActivity(new Intent(ProfileActivity.this, EditProfileActivity.class)));

        // 로그인 / 회원가입 화면 이동 버튼
        MaterialButton btnGoLogin = findViewById(R.id.btn_go_login);
        btnGoLogin.setOnClickListener(v ->
                startActivity(new Intent(ProfileActivity.this, LoginActivity.class)));

        MaterialButton btnGoRegister = findViewById(R.id.btn_go_register);
        btnGoRegister.setOnClickListener(v ->
                startActivity(new Intent(ProfileActivity.this, RegisterActivity.class)));
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateUI();
    }

    private void updateUI() {
        FirebaseUser user = mAuth.getCurrentUser();

        if (user != null) {
            layoutLoggedIn.setVisibility(View.VISIBLE);
            layoutLoggedOut.setVisibility(View.GONE);

            TextView tvName  = findViewById(R.id.tv_user_name);
            TextView tvEmail = findViewById(R.id.tv_user_email);

            // Auth 기본 정보 표시 (Firestore 로드 전 fallback)
            String authName = user.getDisplayName();
            tvName.setText((authName != null && !authName.isEmpty()) ? authName : "이름 없음");
            tvEmail.setText(user.getEmail() != null ? user.getEmail() : "이메일 없음");

            // Firestore에서 유저 상세 정보 로드
            loadUserFromFirestore(user.getUid());

        } else {
            layoutLoggedIn.setVisibility(View.GONE);
            layoutLoggedOut.setVisibility(View.VISIBLE);
        }
    }

    private void loadUserFromFirestore(String uid) {
        db.collection("users").document(uid)
                .get()
                .addOnSuccessListener(this, document -> {
                    if (document != null && document.exists()) {
                        applyUserDataToUI(document);
                    }
                })
                .addOnFailureListener(e -> {
                    // 실패 시 기본값 유지 (Auth 정보만 표시)
                });
    }

    private void applyUserDataToUI(DocumentSnapshot document) {
        TextView tvName  = findViewById(R.id.tv_user_name);
        TextView tvEmail = findViewById(R.id.tv_user_email);

        // 이름 / 이메일
        String name  = document.getString("name");
        String email = document.getString("email");
        if (name  != null && !name.isEmpty())  tvName.setText(name);
        if (email != null && !email.isEmpty()) tvEmail.setText(email);

        // 차량 정보
        TextView tvCompact  = findViewById(R.id.tv_compact_car);
        TextView tvElectric = findViewById(R.id.tv_electric_car);
        TextView tvDisabled = findViewById(R.id.tv_disabled);
        TextView tvNational = findViewById(R.id.tv_national_merit);
        TextView tvFemale   = findViewById(R.id.tv_female_parking);

        // carInfo 중첩 객체 읽기
        Boolean isCompact   = document.getBoolean("carInfo.isCompactCar");
        Boolean isElectric  = document.getBoolean("carInfo.isElectricCar");
        Boolean isDisabled  = document.getBoolean("carInfo.isDisabled");
        Boolean isNational  = document.getBoolean("carInfo.isNationalMeritRecipient");
        Boolean isFemale    = document.getBoolean("carInfo.canUseFemaleParking");

        // Map 경로를 직접 읽을 때 getBoolean("a.b") 가 null 반환할 수 있으므로 안전하게 처리
        if (isCompact  == null) isCompact  = getNestedBoolean(document, "carInfo", "isCompactCar");
        if (isElectric == null) isElectric = getNestedBoolean(document, "carInfo", "isElectricCar");
        if (isDisabled == null) isDisabled = getNestedBoolean(document, "carInfo", "isDisabled");
        if (isNational == null) isNational = getNestedBoolean(document, "carInfo", "isNationalMeritRecipient");
        if (isFemale   == null) isFemale   = getNestedBoolean(document, "carInfo", "canUseFemaleParking");

        tvCompact.setText("경차: "          + boolToText(isCompact));
        tvElectric.setText("전기차: "        + boolToText(isElectric));
        tvDisabled.setText("장애인 차량: "   + boolToText(isDisabled));
        tvNational.setText("국가유공자: "    + boolToText(isNational));
        tvFemale.setText("여성배려주차: "     + boolToText(isFemale));
    }

    /** Firestore 중첩 Map에서 Boolean 안전하게 읽기 */
    @SuppressWarnings("unchecked")
    private Boolean getNestedBoolean(DocumentSnapshot doc, String mapKey, String fieldKey) {
        Object mapObj = doc.get(mapKey);
        if (mapObj instanceof java.util.Map) {
            java.util.Map<String, Object> map = (java.util.Map<String, Object>) mapObj;
            Object val = map.get(fieldKey);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return false;
    }

    private String boolToText(Boolean value) {
        if (value == null) return "-";
        return value ? "✓ 해당" : "해당 없음";
    }
}
