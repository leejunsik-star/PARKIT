package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.UserProfileChangeRequest;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Map;

public class RegisterActivity extends AppCompatActivity {

    private TextInputEditText etName, etEmail, etPassword, etPasswordConfirm;
    private CheckBox cbCompactCar, cbElectricCar, cbDisabled, cbNationalMerit, cbFemaleParking;
    private FirebaseAuth mAuth;
    private FirebaseFirestore db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        mAuth = FirebaseAuth.getInstance();
        db    = FirebaseFirestore.getInstance();

        etName            = findViewById(R.id.et_name);
        etEmail           = findViewById(R.id.et_email);
        etPassword        = findViewById(R.id.et_password);
        etPasswordConfirm = findViewById(R.id.et_password_confirm);

        cbCompactCar   = findViewById(R.id.cb_compact_car);
        cbElectricCar  = findViewById(R.id.cb_electric_car);
        cbDisabled     = findViewById(R.id.cb_disabled);
        cbNationalMerit= findViewById(R.id.cb_national_merit);
        cbFemaleParking= findViewById(R.id.cb_female_parking);

        Button btnRegister    = findViewById(R.id.btn_register);
        Button btnBackToLogin = findViewById(R.id.btn_back_to_login);

        btnRegister.setOnClickListener(v -> performRegister());
        btnBackToLogin.setOnClickListener(v -> finish());
    }

    private void performRegister() {
        String name     = etName.getText() != null ? etName.getText().toString().trim() : "";
        String email    = etEmail.getText() != null ? etEmail.getText().toString().trim() : "";
        String password = etPassword.getText() != null ? etPassword.getText().toString().trim() : "";
        String confirm  = etPasswordConfirm.getText() != null ? etPasswordConfirm.getText().toString().trim() : "";

        if (TextUtils.isEmpty(name)) {
            etName.setError("이름을 입력하세요");
            return;
        }
        if (TextUtils.isEmpty(email)) {
            etEmail.setError("이메일을 입력하세요");
            return;
        }
        if (password.length() < 6) {
            etPassword.setError("비밀번호는 6자 이상이어야 합니다");
            return;
        }
        if (!password.equals(confirm)) {
            etPasswordConfirm.setError("비밀번호가 일치하지 않습니다");
            return;
        }

        mAuth.createUserWithEmailAndPassword(email, password)
                .addOnCompleteListener(this, task -> {
                    if (task.isSuccessful() && mAuth.getCurrentUser() != null) {
                        String uid = mAuth.getCurrentUser().getUid();

                        // Firebase Auth displayName 저장
                        UserProfileChangeRequest profileUpdates = new UserProfileChangeRequest.Builder()
                                .setDisplayName(name)
                                .build();
                        mAuth.getCurrentUser().updateProfile(profileUpdates);

                        // Firestore에 유저 정보 저장
                        saveUserToFirestore(uid, name, email);

                    } else {
                        String msg = task.getException() != null
                                ? task.getException().getMessage()
                                : "회원가입에 실패했습니다.";
                        Toast.makeText(RegisterActivity.this, "오류: " + msg, Toast.LENGTH_LONG).show();
                    }
                });
    }

    private void saveUserToFirestore(String uid, String name, String email) {
        // carInfo Map 구성
        Map<String, Object> carInfo = new HashMap<>();
        carInfo.put("isCompactCar",            cbCompactCar.isChecked());
        carInfo.put("isElectricCar",           cbElectricCar.isChecked());
        carInfo.put("isDisabled",              cbDisabled.isChecked());
        carInfo.put("isNationalMeritRecipient",cbNationalMerit.isChecked());
        carInfo.put("canUseFemaleParking",     cbFemaleParking.isChecked());

        // users 컬렉션 문서 구성
        Map<String, Object> user = new HashMap<>();
        user.put("name",    name);
        user.put("email",   email);
        user.put("carInfo", carInfo);

        db.collection("users").document(uid)
                .set(user)
                .addOnSuccessListener(aVoid -> {
                    // Firestore 저장 완료 후 로그아웃 → 로그인 화면으로
                    mAuth.signOut();
                    Toast.makeText(RegisterActivity.this,
                            "회원가입이 완료되었습니다! 로그인해주세요.", Toast.LENGTH_SHORT).show();
                    finish();
                })
                .addOnFailureListener(e -> {
                    // Firestore 저장 실패 시에도 Auth 계정은 생성됐으므로 안내 후 진행
                    mAuth.signOut();
                    Toast.makeText(RegisterActivity.this,
                            "회원가입 완료 (차량 정보 저장 실패: " + e.getMessage() + ")",
                            Toast.LENGTH_LONG).show();
                    finish();
                });
    }
}
