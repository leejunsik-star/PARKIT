package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Typeface;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import android.text.StaticLayout;
import android.text.TextPaint;
import android.text.Layout;
import android.os.Build;

/**
 * 주차장 말풍선 마커 Bitmap을 Canvas로 생성하는 유틸리티 클래스.
 *
 * 마커 구조:
 * ┌──────────────────────┐
 * │          30          │  ← 숫자 (잔여율 색상 배경)
 * ├──────────┬───────────┤
 * │  공영    │   유료    │  ← 하단 좌/우
 * └──────────┴───────────┘
 *            ▼  ← 말풍선 꼬리
 */
public class ParkingMarkerHelper {

    // ──────────────────────────────────────────────────────────────────
    // 공개 API
    // ──────────────────────────────────────────────────────────────────

    /**
     * 말풍선 마커 Bitmap 생성
     *
     * @param ctx     Context (dp→px 변환용)
     * @param spot    주차장 데이터
     * @param carInfo 사용자 차량 자격 (비회원이면 isGuest=true)
     * @return        카카오맵 Label에 사용할 Bitmap
     */
    public static Bitmap createMarkerBitmap(Context ctx, ParkingSpot spot, UserCarInfo carInfo) {
        int[] calc      = calculateRemaining(spot, carInfo);
        int remaining   = calc[0];
        int total       = calc[1];
        int accentColor = getCountColor(remaining, total);

        float density = ctx.getResources().getDisplayMetrics().density;
        int bodyW   = dp(density, 100);  // 전체 너비 (100dp)
        int iconH   = dp(density, 16);   // 1단: 카테고리 아이콘 영역 높이 (16dp)
        int bottomH = dp(density, 14);   // 3단: 하단 정보 영역 높이 (14dp)

        // ── 1. 주차장 이름 여러 줄 줄바꿈 크기(높이) 실시간 계산 ──
        TextPaint namePaint = new TextPaint(Paint.ANTI_ALIAS_FLAG);
        namePaint.setColor(Color.parseColor("#222222")); // 지도 배경과 분리되는 선명한 회색
        namePaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        namePaint.setTextSize(dp(density, 9.5f));

        String displayName = spot.name != null ? spot.name : "주차장";
        int textWidth = dp(density, 90); // 여백 제외 텍스트 가로 너비
        StaticLayout staticLayout;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            staticLayout = StaticLayout.Builder.obtain(displayName, 0, displayName.length(), namePaint, textWidth)
                    .setAlignment(Layout.Alignment.ALIGN_CENTER)
                    .setLineSpacing(0f, 1.0f)
                    .setIncludePad(false)
                    .build();
        } else {
            staticLayout = new StaticLayout(displayName, namePaint, textWidth,
                    Layout.Alignment.ALIGN_CENTER, 1.0f, 0f, false);
        }
        int nameH = staticLayout.getHeight(); // 텍스트 줄 수에 따른 동적 높이

        // 3dp 패딩 공간(그림자 가림 방지) 추가
        int padding = dp(density, 3);
        int contentGap = dp(density, 4); // 이름과 배지 간 여백
        int bodyH   = padding * 2 + nameH + bottomH + contentGap;
        int totalH  = bodyH;

        Bitmap bmp    = Bitmap.createBitmap(bodyW, totalH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmp);

        Paint fill   = new Paint(Paint.ANTI_ALIAS_FLAG);

        // ── 2. 지도 배경과의 식별을 위한 Glassmorphism(반투명) 입체 카드 배경 ──
        Paint bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        bgPaint.setColor(Color.parseColor("#E6FFFFFF")); // 90% 투명도 흰색
        bgPaint.setShadowLayer(dp(density, 2.5f), 0f, dp(density, 1f), Color.parseColor("#25000000")); // 가벼운 그림자

        RectF bgRect = new RectF(padding, padding, bodyW - padding, bodyH - padding);
        canvas.drawRoundRect(bgRect, dp(density, 8), dp(density, 8), bgPaint);

        // ── 3. 1단: 상단에 주차장 이름 그리기 (여러 줄 지원) ──
        canvas.save();
        // 가로 5dp 여백(textWidth = 90dp이므로 100dp의 중앙에 맞춤)
        canvas.translate(dp(density, 5), padding + dp(density, 2));
        staticLayout.draw(canvas);
        canvas.restore();

        // ── 4. 2단: 하단 가로 밀집 배치 (3개 배지 모음 정렬) ──
        float bottomCenterY = padding + nameH + contentGap + bottomH / 2f;

        // 배지 총 합산 가로폭: 58dp (잔여석 28dp + 간격 4dp + 유형 11dp + 간격 4dp + 요금 11dp)
        float startX = (bodyW - dp(density, 58)) / 2f;

        // 4-1. 왼쪽 배지: 잔여율 색상 배경의 둥근 사각형 배지 + 남은 대수 텍스트
        float badgeW = dp(density, 28);      // 배지 너비
        float badgeH = dp(density, 11);      // 배지 높이
        float badgeCenterX = startX + badgeW / 2f;

        RectF badgeRect = new RectF(
                badgeCenterX - badgeW / 2f,
                bottomCenterY - badgeH / 2f,
                badgeCenterX + badgeW / 2f,
                bottomCenterY + badgeH / 2f
        );
        fill.setColor(accentColor);
        canvas.drawRoundRect(badgeRect, dp(density, 3), dp(density, 3), fill);

        Paint badgeTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        badgeTextPaint.setColor(Color.WHITE);
        badgeTextPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        badgeTextPaint.setTextAlign(Paint.Align.CENTER);
        badgeTextPaint.setTextSize(dp(density, 7.5f));

        String remainingText = remaining + "대";
        float badgeTextY = bottomCenterY - (badgeTextPaint.descent() + badgeTextPaint.ascent()) / 2f;
        canvas.drawText(remainingText, badgeCenterX, badgeTextY, badgeTextPaint);

        // 4-2. 중앙 배지: 카테고리(공영/민영/상가) 원형 아이콘 배지
        float radius = dp(density, 5.5f);
        float categoryCenterX = startX + badgeW + dp(density, 4) + radius;
        drawCategoryBadge(canvas, categoryCenterX, bottomCenterY, radius, spot.category, density);

        // 4-3. 오른쪽 배지: 요금 유형 원형 아이콘 배지 (₩, F, ?)
        float feeCenterX = startX + badgeW + dp(density, 4) + radius * 2f + dp(density, 4) + radius;
        drawFeeBadge(canvas, feeCenterX, bottomCenterY, radius, spot.feeType, density);

        return bmp;
    }

    // ──────────────────────────────────────────────────────────────────
    // 계산 유틸리티
    // ──────────────────────────────────────────────────────────────────

    /**
     * carInfo에 따라 사용 가능한 구역의 (남은 자리, 전체 자리) 반환
     * @return int[]{ remaining, total }
     */
    @SuppressWarnings("unchecked")
    public static int[] calculateRemaining(ParkingSpot spot, UserCarInfo carInfo) {
        if (spot.parkingLots == null) return new int[]{0, 0};

        String[] keys;
        if (carInfo.isGuest) {
            // 비회원: 오직 general 구역만 조회
            keys = new String[]{"general"};
        } else {
            // 로그인: general + 해당 자격 구역 합산
            List<String> keyList = new ArrayList<>();
            keyList.add("general");
            if (carInfo.isCompactCar)             keyList.add("compact");
            if (carInfo.isElectricCar)            keyList.add("electric");
            if (carInfo.isDisabled)               keyList.add("disabled");
            if (carInfo.isNationalMeritRecipient) keyList.add("National");
            if (carInfo.canUseFemaleParking)      keyList.add("Female");
            keys = keyList.toArray(new String[0]);
        }

        int totalSpots = 0, remaining = 0;
        for (String key : keys) {
            Object zoneObj = spot.parkingLots.get(key);
            if (zoneObj instanceof Map) {
                Map<String, Object> zone = (Map<String, Object>) zoneObj;
                long t = getLong(zone, "total");
                long c = getLong(zone, "current"); // current = 현재 남은 공석 대수
                totalSpots += t;
                remaining  += c; // total - current 가 아니라 current 값을 직접 합산
            }
        }

        return new int[]{remaining, totalSpots};
    }

    /** 잔여율에 따른 배경 색상 반환 */
    public static int getCountColor(int remaining, int total) {
        if (remaining == 0 || total == 0) return Color.parseColor("#9E9E9E"); // 회색
        double ratio = (double) remaining / total;
        if (ratio <= 0.05) return Color.parseColor("#F44336"); // 빨강
        if (ratio <= 0.30) return Color.parseColor("#FFC107"); // 노랑
        return Color.parseColor("#4CAF50");                    // 초록
    }

    // ──────────────────────────────────────────────────────────────────
    // 배지(아이콘) 렌더링 유틸리티
    // ──────────────────────────────────────────────────────────────────

    private static void drawCategoryBadge(Canvas canvas, float cx, float cy, float r, String category, float density) {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        textPaint.setTextSize(dp(density, 6)); // 아이콘 안의 글자 크기

        String text = "P";
        int bgColor = Color.GRAY;
        int textColor = Color.WHITE;

        if (category == null) category = "기타";
        switch (category) {
            case "public":
                bgColor = Color.parseColor("#2196F3"); // 파란색
                break;
            case "private":
                bgColor = Color.parseColor("#FFC107"); // 노란색
                textColor = Color.parseColor("#333333");
                break;
            case "business":
                bgColor = Color.parseColor("#4CAF50"); // 초록색
                text = "🏢";
                textPaint.setTextSize(dp(density, 5)); // 이모지는 조금 작게
                break;
        }

        // 배경 원 그리기
        paint.setColor(bgColor);
        canvas.drawCircle(cx, cy, r, paint);

        // 텍스트(아이콘) 그리기
        textPaint.setColor(textColor);
        float textY = cy - (textPaint.descent() + textPaint.ascent()) / 2f;
        canvas.drawText(text, cx, textY, textPaint);
    }

    private static void drawFeeBadge(Canvas canvas, float cx, float cy, float r, String feeType, float density) {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        textPaint.setTextSize(dp(density, 6));

        String text = "?";
        int bgColor = Color.parseColor("#9E9E9E"); // 기본 회색 (조건부 무료 포함)
        int textColor = Color.WHITE;

        if (feeType == null) feeType = "-";
        switch (feeType) {
            case "paid":
                bgColor = Color.parseColor("#FF9800"); // 금색/주황색
                text = "₩";
                break;
            case "free":
                bgColor = Color.parseColor("#4CAF50"); // 초록색
                text = "F";
                break;
            case "conditional_free":
            case "conditional free": // DB 오타 방어 코드
                bgColor = Color.parseColor("#9E9E9E"); // 회색
                text = "?";
                break;
        }

        // 배경 원 그리기
        paint.setColor(bgColor);
        canvas.drawCircle(cx, cy, r, paint);

        // 텍스트 그리기
        textPaint.setColor(textColor);
        float textY = cy - (textPaint.descent() + textPaint.ascent()) / 2f;
        canvas.drawText(text, cx, textY, textPaint);
    }

    // ──────────────────────────────────────────────────────────────────
    // 내부 드로잉 유틸리티
    // ──────────────────────────────────────────────────────────────────

    /** 말풍선 전체 Path (rounded rect + 아래 삼각 꼬리) */
    private static Path buildBubblePath(int bodyW, int bodyH, int tailH, int corner) {
        Path path = new Path();
        RectF rect = new RectF(0, 0, bodyW, bodyH);
        path.addRoundRect(rect, corner, corner, Path.Direction.CW);

        // 삼각 꼬리 (가운데 아래)
        int cx = bodyW / 2;
        path.moveTo(cx - tailH, bodyH);
        path.lineTo(cx, bodyH + tailH);
        path.lineTo(cx + tailH, bodyH);
        path.close();
        return path;
    }

    /** 상단 색상 영역 Path (위쪽 모서리만 둥글게) */
    private static Path buildTopSectionPath(int bodyW, int topH, int corner) {
        Path path = new Path();
        float[] radii = {corner, corner, corner, corner, 0, 0, 0, 0};
        RectF rect = new RectF(0, 0, bodyW, topH);
        path.addRoundRect(rect, radii, Path.Direction.CW);
        return path;
    }

    /** dp → px 변환 */
    private static int dp(float density, float dp) {
        return Math.round(dp * density);
    }

    /** Map에서 Long 안전하게 읽기 */
    private static long getLong(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof Long)    return (Long) val;
        if (val instanceof Integer) return ((Integer) val).longValue();
        return 0;
    }
}
