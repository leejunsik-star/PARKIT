package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.Executors;

public class ParkingMapView extends View {

    private ParkingSpot.Zone zoneData;
    private String currentParkingId;
    private Bitmap bgBitmap;
    private UserCarInfo currentCarInfo;
    private boolean isFitCentered = false;

    // 내 주차 차량 자리 정보
    private String myParkedParkingId;
    private String myParkedSpotId;

    public interface OnSpotClickListener {
        void onSpotClick(ParkingSpot.Zone zone, ParkingSpot.Spot spot);
    }
    private OnSpotClickListener spotClickListener;

    public void setMyParkedInfo(String parkingId, String spotId) {
        this.myParkedParkingId = parkingId;
        this.myParkedSpotId = spotId;
        invalidate();
    }

    public void setOnSpotClickListener(OnSpotClickListener listener) {
        this.spotClickListener = listener;
    }

    // 이미지 메모리 캐시
    private static final java.util.HashMap<String, Bitmap> bgCache = new java.util.HashMap<>();

    // ── 드래그 및 줌인/줌아웃 인터랙티브 상태 ──
    private float scaleFactor = 1.0f;
    private float posX = 0f;
    private float posY = 0f;
    private float lastTouchX;
    private float lastTouchY;
    private int activePointerId = MotionEvent.INVALID_POINTER_ID;

    private ScaleGestureDetector scaleGestureDetector;
    private GestureDetector gestureDetector;

    public void setCurrentCarInfo(UserCarInfo carInfo) {
        this.currentCarInfo = carInfo;
        invalidate();
    }

    public ParkingMapView(Context context) {
        super(context);
        init(context);
    }

    public ParkingMapView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    public ParkingMapView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init(context);
    }

    private void init(Context context) {
        // 줌 제스처 감지기 초기화
        scaleGestureDetector = new ScaleGestureDetector(context, new ScaleListener());
        
        // 더블탭 시 줌 초기화하는 더블탭 감지기 및 싱글탭 터치 감지기
        gestureDetector = new GestureDetector(context, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onDoubleTap(MotionEvent e) {
                scaleFactor = 1.0f;
                posX = 0f;
                posY = 0f;
                invalidate();
                return true;
            }

            @Override
            public boolean onSingleTapConfirmed(MotionEvent e) {
                checkSpotClicked(e.getX(), e.getY());
                return true;
            }
        });
    }

    private void checkSpotClicked(float touchX, float touchY) {
        if (zoneData == null || spotClickListener == null) return;

        // 줌 및 팬 스크롤 위치 역산
        float canvasX = (touchX - posX) / scaleFactor;
        float canvasY = (touchY - posY) / scaleFactor;

        if (zoneData.spots != null && zoneData.groups != null) {
            for (ParkingSpot.Spot spot : zoneData.spots) {
                ParkingSpot.Group group = findGroup(spot.groupId);
                if (group == null) continue;

                int[] spec = getSpotSpec(spot.type);
                float width = spec[0];
                float height = spec[1];

                // 2D 회전 역변환 (Group rotation 역적용)
                double dx = canvasX - group.x;
                double dy = canvasY - group.y;
                double rad = (-group.rotation * Math.PI) / 180.0;
                double rx = dx * Math.cos(rad) - dy * Math.sin(rad);
                double ry = dx * Math.sin(rad) + dy * Math.cos(rad);

                // 사각형 AABB 충돌 검사
                if (rx >= spot.localX && rx <= spot.localX + width &&
                    ry >= spot.localY && ry <= spot.localY + height) {
                    
                    // 주차칸 터치 콜백 트리거
                    spotClickListener.onSpotClick(zoneData, spot);
                    break;
                }
            }
        }
    }

    /**
     * Firestore로부터 취득한 단일 Zone 도면 데이터를 주입합니다.
     */
    public void setZoneData(ParkingSpot.Zone zone, String parkingId) {
        this.zoneData = zone;
        this.currentParkingId = parkingId;
        this.bgBitmap = null;
        this.scaleFactor = 1.0f;
        this.posX = 0f;
        this.posY = 0f;
        this.isFitCentered = false;
        
        if (zone != null) {
            loadBackgroundImage(zone.bgImgEmpty);
        }
        invalidate();
    }

    /**
     * 배경 이미지를 백그라운드 스레드에서 다운로드합니다.
     */
    private void loadBackgroundImage(String urlStr) {
        if (urlStr == null || urlStr.isEmpty()) return;
        if (bgCache.containsKey(urlStr)) {
            bgBitmap = bgCache.get(urlStr);
            invalidate();
            return;
        }
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                URL url = new URL(urlStr);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setDoInput(true);
                conn.connect();
                InputStream input = conn.getInputStream();
                Bitmap bitmap = BitmapFactory.decodeStream(input);
                post(() -> {
                    bgBitmap = bitmap;
                    bgCache.put(urlStr, bitmap);
                    invalidate();
                });
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (zoneData == null) return;

        // ── 초기 1회 도면 화면 중앙 정렬 및 Fit ──
        if (bgBitmap != null && getWidth() > 0 && getHeight() > 0 && !isFitCentered) {
            float bgW = bgBitmap.getWidth() * zoneData.bgScaleX;
            float bgH = bgBitmap.getHeight() * zoneData.bgScaleY;

            float scaleX = (float) getWidth() / (bgW > 0 ? bgW : 1f);
            float scaleY = (float) getHeight() / (bgH > 0 ? bgH : 1f);
            scaleFactor = Math.min(scaleX, scaleY) * 0.9f; // 10% 여유

            float centerX = zoneData.bgPosX + bgW / 2f;
            float centerY = zoneData.bgPosY + bgH / 2f;

            posX = getWidth() / 2f - centerX * scaleFactor;
            posY = getHeight() / 2f - centerY * scaleFactor;

            isFitCentered = true;
        }

        canvas.save();
        
        // ── 1. 멀티터치 드래그 및 줌인 변환 적용 ──
        canvas.translate(posX, posY);
        canvas.scale(scaleFactor, scaleFactor);

        // ── 2. 배경 이미지 그리기 ──
        if (bgBitmap != null) {
            float bgW = bgBitmap.getWidth() * zoneData.bgScaleX;
            float bgH = bgBitmap.getHeight() * zoneData.bgScaleY;
            canvas.drawBitmap(bgBitmap, null, new RectF(zoneData.bgPosX, zoneData.bgPosY, zoneData.bgPosX + bgW, zoneData.bgPosY + bgH), null);
        }

        // ── 3. 실시간 주차칸 렌더링 ──
        if (zoneData.spots != null && zoneData.groups != null) {
            for (ParkingSpot.Spot spot : zoneData.spots) {
                // 해당 주차칸의 그룹 매칭
                ParkingSpot.Group group = findGroup(spot.groupId);
                if (group == null) continue;

                // 주차칸 스펙 가져오기 (가로, 세로, 선 색상, 채우기 색상)
                int[] spec = getSpotSpec(spot.type);
                float width = spec[0];
                float height = spec[1];
                int strokeColor = spec[2];
                int fillColor = spec[3];

                // 사용자의 해당 주차구획 자격 여부 검사
                boolean isEligible = true;
                boolean isMyCarSpot = myParkedParkingId != null && myParkedParkingId.equals(currentParkingId) && myParkedSpotId != null && myParkedSpotId.equals(spot.id);

                if (currentCarInfo != null) {
                    if (currentCarInfo.isGuest) {
                        // 비회원은 일반 구역(NORMAL)만 이용 가능
                        if (!"NORMAL".equals(spot.type)) {
                            isEligible = false;
                        }
                    } else {
                        // 로그인 회원은 정보 확인
                        if ("COMPACT".equals(spot.type) && !currentCarInfo.isCompactCar) {
                            isEligible = false;
                        } else if ("EV".equals(spot.type) && !currentCarInfo.isElectricCar) {
                            isEligible = false;
                        } else if ("HANDICAP".equals(spot.type) && !currentCarInfo.isDisabled) {
                            isEligible = false;
                        } else if ("PATRIOT".equals(spot.type) && !currentCarInfo.isNationalMeritRecipient) {
                            isEligible = false;
                        } else if ("WOMAN".equals(spot.type) && !currentCarInfo.canUseFemaleParking) {
                            isEligible = false;
                        }
                    }
                }

                // 2D 회전 매트릭스를 적용해 절대 중심 좌표 산출
                double rad = (group.rotation * Math.PI) / 180.0;
                double cos = Math.cos(rad);
                double sin = Math.sin(rad);

                double cx = spot.localX + width / 2.0;
                double cy = spot.localY + height / 2.0;

                double globalCenterX = group.x + (cx * cos - cy * sin);
                double globalCenterY = group.y + (cx * sin + cy * cos);

                // ── 캔버스 변환 적용하여 개별 회전 그리기 ──
                canvas.save();
                canvas.translate((float) globalCenterX, (float) globalCenterY);
                canvas.rotate(group.rotation);

                // 회전된 좌표계에서의 사각형 범위 (-w/2, -h/2 ~ w/2, h/2)
                RectF rect = new RectF(-width / 2f, -height / 2f, width / 2f, height / 2f);

                // 그리기용 Paint
                Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

                if (isMyCarSpot) {
                    // [내차 주차 상태 시각화]: 골드색 반투명 채우기, 두꺼운 골드 테두리 (자격 유무/점유 여부 불문 최우선)
                    paint.setStyle(Paint.Style.FILL);
                    paint.setColor(Color.parseColor("#4DFFC107")); // rgba(255, 193, 7, 0.30)
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                    paint.setStyle(Paint.Style.STROKE);
                    paint.setColor(Color.parseColor("#FFC107"));
                    paint.setStrokeWidth(dp(3));
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                    // 중앙에 "내차" 텍스트
                    Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
                    textPaint.setColor(Color.parseColor("#2B6CB0")); // 진한 파란색
                    textPaint.setTextSize(dp(10));
                    textPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
                    textPaint.setTextAlign(Paint.Align.CENTER);
                    float textY = -(textPaint.descent() + textPaint.ascent()) / 2f;
                    canvas.drawText("내차", 0f, textY, textPaint);

                } else if (!isEligible) {
                    // [자격 미해당 시각화]: 회색조 비활성화 처리 (점유 여부 불문)
                    paint.setStyle(Paint.Style.FILL);
                    paint.setColor(Color.parseColor("#1AE2E8F0"));
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                    paint.setStyle(Paint.Style.STROKE);
                    paint.setColor(Color.parseColor("#A0AEC0"));
                    paint.setStrokeWidth(dp(2));
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                    // 빗금질(Hatching) 효과 추가
                    canvas.save();
                    android.graphics.Path clipPath = new android.graphics.Path();
                    clipPath.addRoundRect(rect, dp(4), dp(4), android.graphics.Path.Direction.CW);
                    canvas.clipPath(clipPath);

                    Paint hatchPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
                    hatchPaint.setColor(Color.parseColor("#4DA0AEC0")); // 빗금의 불투명도를 살짝 올림 (30%)
                    hatchPaint.setStrokeWidth(dp(1.5f));
                    
                    float step = dp(6); // 빗금 간격
                    for (float offset = -width - height; offset < width + height; offset += step) {
                        canvas.drawLine(-width / 2f + offset, -height / 2f, -width / 2f + offset + height, height / 2f, hatchPaint);
                    }
                    canvas.restore();

                } else if (spot.occupied) {
                    // [점유 상태 시각화]: 빨간색 반투명 채우기, 두꺼운 테두리
                    paint.setStyle(Paint.Style.FILL);
                    paint.setColor(Color.parseColor("#73E53E3E")); // rgba(229, 62, 62, 0.45) ≒ 73 (Hex alpha)
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                    paint.setStyle(Paint.Style.STROKE);
                    paint.setColor(Color.parseColor("#E53E3E"));
                    paint.setStrokeWidth(dp(3));
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                } else {
                    // [비점유 상태 시각화]: 기본 타입별 색상 적용, opacity 적용
                    float opacity = zoneData.spotOpacity;

                    paint.setStyle(Paint.Style.FILL);
                    paint.setColor(applyOpacity(fillColor, opacity));
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);

                    paint.setStyle(Paint.Style.STROKE);
                    paint.setColor(applyOpacity(strokeColor, opacity));
                    paint.setStrokeWidth(dp(2));
                    canvas.drawRoundRect(rect, dp(4), dp(4), paint);
                }

                canvas.restore();
            }
        }

        canvas.restore();
    }

    private ParkingSpot.Group findGroup(String groupId) {
        if (zoneData.groups == null || groupId == null) return null;
        for (ParkingSpot.Group g : zoneData.groups) {
            if (groupId.equals(g.id)) {
                return g;
            }
        }
        return null;
    }

    /**
     * 주차칸 타입별 스펙 반환
     * @return int[]{ width, height, strokeColor, fillColor }
     */
    private int[] getSpotSpec(String type) {
        if (type == null) type = "NORMAL";
        int w = 40, h = 80;
        int stroke = Color.parseColor("#cbd5e0");
        int fill = Color.parseColor("#ffffff");

        switch (type) {
            case "COMPACT":
                w = 35; h = 70;
                stroke = Color.parseColor("#38a169");
                fill = Color.parseColor("#4048bb78"); // rgba(72,187,120,0.25)
                break;
            case "EV":
                w = 40; h = 80;
                stroke = Color.parseColor("#d69e2e");
                fill = Color.parseColor("#40ecc94b"); // rgba(236,201,75,0.25)
                break;
            case "HANDICAP":
                w = 55; h = 80;
                stroke = Color.parseColor("#2b6cb0");
                fill = Color.parseColor("#4D3182ce"); // rgba(49,130,206,0.30)
                break;
            case "PATRIOT":
                w = 40; h = 80;
                stroke = Color.parseColor("#805ad5");
                fill = Color.parseColor("#409f7ae2"); // rgba(159,122,234,0.25)
                break;
            case "WOMAN":
                w = 40; h = 80;
                stroke = Color.parseColor("#d53f8c");
                fill = Color.parseColor("#40f687b3"); // rgba(246,135,179,0.25)
                break;
            case "NORMAL":
            default:
                w = 40; h = 80;
                stroke = Color.parseColor("#cbd5e0");
                fill = Color.WHITE;
                break;
        }

        return new int[]{ w, h, stroke, fill };
    }

    private int applyOpacity(int color, float opacity) {
        int alpha = Math.round(Color.alpha(color) * opacity);
        return Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }

    // ── 터치 이벤트 핸들링 (줌 및 드래그 제어) ──

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        // 더블탭 이벤트 위임
        gestureDetector.onTouchEvent(event);
        // 멀티터치 스케일 이벤트 위임
        scaleGestureDetector.onTouchEvent(event);

        final int action = event.getActionMasked();
        switch (action) {
            case MotionEvent.ACTION_DOWN: {
                final float x = event.getX();
                final float y = event.getY();
                lastTouchX = x;
                lastTouchY = y;
                activePointerId = event.getPointerId(0);
                break;
            }
            case MotionEvent.ACTION_MOVE: {
                final int pointerIndex = event.findPointerIndex(activePointerId);
                if (pointerIndex != MotionEvent.INVALID_POINTER_ID) {
                    final float x = event.getX(pointerIndex);
                    final float y = event.getY(pointerIndex);

                    if (!scaleGestureDetector.isInProgress()) {
                        final float dx = x - lastTouchX;
                        final float dy = y - lastTouchY;
                        posX += dx;
                        posY += dy;
                        constrainPanAndZoom();
                        invalidate();
                    }

                    lastTouchX = x;
                    lastTouchY = y;
                }
                break;
            }
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL: {
                activePointerId = MotionEvent.INVALID_POINTER_ID;
                break;
            }
            case MotionEvent.ACTION_POINTER_UP: {
                final int pointerIndex = event.getActionIndex();
                final int pointerId = event.getPointerId(pointerIndex);
                if (pointerId == activePointerId) {
                    final int newPointerIndex = pointerIndex == 0 ? 1 : 0;
                    lastTouchX = event.getX(newPointerIndex);
                    lastTouchY = event.getY(newPointerIndex);
                    activePointerId = event.getPointerId(newPointerIndex);
                }
                break;
            }
        }
        return true;
    }

    private void constrainPanAndZoom() {
        if (bgBitmap == null || zoneData == null) return;
        float bgW = bgBitmap.getWidth() * zoneData.bgScaleX;
        float bgH = bgBitmap.getHeight() * zoneData.bgScaleY;

        float minScale = Math.min((float) getWidth() / (bgW > 0 ? bgW : 1f), (float) getHeight() / (bgH > 0 ? bgH : 1f)) * 0.5f;
        if (minScale <= 0f) minScale = 0.3f;
        scaleFactor = Math.max(minScale, Math.min(scaleFactor, 5.0f));

        float viewW = getWidth();
        float viewH = getHeight();

        float minX = viewW * 0.1f - (zoneData.bgPosX + bgW) * scaleFactor;
        float maxX = viewW * 0.9f - zoneData.bgPosX * scaleFactor;
        float minY = viewH * 0.1f - (zoneData.bgPosY + bgH) * scaleFactor;
        float maxY = viewH * 0.9f - zoneData.bgPosY * scaleFactor;

        if (minX > maxX) {
            float temp = minX;
            minX = maxX;
            maxX = temp;
        }
        if (minY > maxY) {
            float temp = minY;
            minY = maxY;
            maxY = temp;
        }

        posX = Math.max(minX, Math.min(posX, maxX));
        posY = Math.max(minY, Math.min(posY, maxY));
    }

    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            scaleFactor *= detector.getScaleFactor();
            constrainPanAndZoom();
            invalidate();
            return true;
        }
    }
}
