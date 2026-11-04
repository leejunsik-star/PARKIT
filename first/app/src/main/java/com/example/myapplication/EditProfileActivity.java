package com.example.myapplication;

import android.os.Bundle;
import android.text.TextUtils;
import android.widget.CheckBox;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.auth.UserProfileChangeRequest;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Map;

public class EditProfileActivity extends AppCompatActivity {

    private TextInputEditText etEditName;
    private CheckBox cbCompactCar, cbElectricCar, cbDisabled, cbNationalMerit, cbFemaleParking;
    private FirebaseAuth mAuth;
    private FirebaseFirestore db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_edit_profile);

        mAuth = FirebaseAuth.getInstance();
        db    = FirebaseFirestore.getInstance();

        etEditName      = findViewById(R.id.et_edit_name);
        cbCompactCar    = findViewById(R.id.cb_edit_compact_car);
        cbElectricCar   = findViewById(R.id.cb_edit_electric_car);
        cbDisabled      = findViewById(R.id.cb_edit_disabled);
        cbNationalMerit = findViewById(R.id.cb_edit_national_merit);
        cbFemaleParking = findViewById(R.id.cb_edit_female_parking);

        MaterialButton btnSave   = findViewById(R.id.btn_save_profile);
        MaterialButton btnCancel = findViewById(R.id.btn_cancel_edit);

        btnSave.setOnClickListener(v -> saveProfile());
        btnCancel.setOnClickListener(v -> finish());

        // 현재 Firestore 데이터 불러와서 폼 초기화
        loadCurrentData();
    }

    private void loadCurrentData() {
        FirebaseUser user = mAuth.getCurrentUser();
        if (user == null) { finish(); return; }

        db.collection("users").document(user.getUid())
                .get()
                .addOnSuccessListener(document -> {
                    if (document != null && document.exists()) {
                        // 이름
                        String name = document.getString("name");
                        if (name != null) etEditName.setText(name);

                        // 차량 정보 (중첩 Map)
                        cbCompactCar.setChecked(   getBool(document, "isCompactCar"));
                        cbElectricCar.setChecked(  getBool(document, "isElectricCar"));
                        cbDisabled.setChecked(     getBool(document, "isDisabled"));
                        cbNationalMerit.setChecked(getBool(document, "isNationalMeritRecipient"));
                        cbFemaleParking.setChecked(getBool(document, "canUseFemaleParking"));
                    }
                })
                .addOnFailureListener(e ->
                        Toast.makeText(this, "데이터 불러오기 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show());
    }

    /** carInfo 중첩 Map에서 Boolean 안전하게 읽기 */
    @SuppressWarnings("unchecked")
    private boolean getBool(DocumentSnapshot doc, String fieldKey) {
        Object mapObj = doc.get("carInfo");
        if (mapObj instanceof Map) {
            Map<String, Object> carInfo = (Map<String, Object>) mapObj;
            Object val = carInfo.get(fieldKey);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return false;
    }

    private void saveProfile() {
        String name = etEditName.getText() != null ? etEditName.getText().toString().trim() : "";
        if (TextUtils.isEmpty(name)) {
            etEditName.setError("이름을 입력하세요");
            return;
        }

        FirebaseUser user = mAuth.getCurrentUser();
        if (user == null) { finish(); return; }

        // carInfo Map 구성
        Map<String, Object> carInfo = new HashMap<>();
        carInfo.put("isCompactCar",             cbCompactCar.isChecked());
        carInfo.put("isElectricCar",            cbElectricCar.isChecked());
        carInfo.put("isDisabled",               cbDisabled.isChecked());
        carInfo.put("isNationalMeritRecipient", cbNationalMerit.isChecked());
        carInfo.put("canUseFemaleParking",      cbFemaleParking.isChecked());

        // Firestore 업데이트
        Map<String, Object> updates = new HashMap<>();
        updates.put("name",    name);
        updates.put("carInfo", carInfo);

        db.collection("users").document(user.getUid())
                .update(updates)
                .addOnSuccessListener(aVoid -> {
                    // Firebase Auth displayName도 동기화
                    UserProfileChangeRequest req = new UserProfileChangeRequest.Builder()
                            .setDisplayName(name)
                            .build();
                    user.updateProfile(req);

                    Toast.makeText(this, "정보가 저장되었습니다.", Toast.LENGTH_SHORT).show();
                    finish(); // ProfileActivity로 복귀
                })
                .addOnFailureListener(e ->
                        Toast.makeText(this, "저장 실패: " + e.getMessage(), Toast.LENGTH_LONG).show());
    }
}
