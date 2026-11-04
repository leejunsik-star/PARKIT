package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.textfield.TextInputEditText;
import com.google.firebase.auth.FirebaseAuth;

public class LoginActivity extends AppCompatActivity {

    private TextInputEditText etEmail, etPassword;
    private FirebaseAuth mAuth;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        mAuth = FirebaseAuth.getInstance();

        etEmail    = findViewById(R.id.et_email);
        etPassword = findViewById(R.id.et_password);

        Button btnLogin      = findViewById(R.id.btn_login);
        Button btnGoRegister = findViewById(R.id.btn_go_register);

        btnLogin.setOnClickListener(v -> performLogin());

        btnGoRegister.setOnClickListener(v ->
                startActivity(new Intent(LoginActivity.this, RegisterActivity.class)));
    }

    private void performLogin() {
        String email    = etEmail.getText() != null ? etEmail.getText().toString().trim() : "";
        String password = etPassword.getText() != null ? etPassword.getText().toString().trim() : "";

        if (TextUtils.isEmpty(email)) {
            etEmail.setError("이메일을 입력하세요");
            return;
        }
        if (TextUtils.isEmpty(password)) {
            etPassword.setError("비밀번호를 입력하세요");
            return;
        }

        mAuth.signInWithEmailAndPassword(email, password)
                .addOnCompleteListener(this, task -> {
                    if (task.isSuccessful()) {
                        // 로그인 성공 → 뒤로 돌아가기 (ProfileActivity가 onResume에서 UI 갱신)
                        Toast.makeText(LoginActivity.this, "로그인 성공!", Toast.LENGTH_SHORT).show();
                        finish();
                    } else {
                        String msg = task.getException() != null
                                ? task.getException().getMessage()
                                : "로그인에 실패했습니다.";
                        Toast.makeText(LoginActivity.this, "로그인 실패: " + msg, Toast.LENGTH_LONG).show();
                    }
                });
    }
}
