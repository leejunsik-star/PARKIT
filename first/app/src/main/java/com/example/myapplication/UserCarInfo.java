package com.example.myapplication;

import com.google.firebase.firestore.DocumentSnapshot;

import java.util.Map;

/**
 * 사용자의 차량 자격 정보를 담는 데이터 클래스.
 * Firestore users/{uid}.carInfo 에서 로드하거나, 비회원용 게스트 인스턴스를 생성합니다.
 */
public class UserCarInfo {

    public boolean isCompactCar;
    public boolean isElectricCar;
    public boolean isDisabled;
    public boolean isNationalMeritRecipient;
    public boolean canUseFemaleParking;

    /** 비회원(게스트) 인스턴스 - 전체 구역 합산 표시 */
    public boolean isGuest;

    /** 비회원용 인스턴스 반환 */
    public static UserCarInfo getGuest() {
        UserCarInfo info = new UserCarInfo();
        info.isGuest = true;
        return info;
    }

    /** Firestore 문서에서 carInfo Map을 파싱하여 반환 */
    @SuppressWarnings("unchecked")
    public static UserCarInfo fromDocument(DocumentSnapshot doc) {
        UserCarInfo info = new UserCarInfo();
        info.isGuest = false;

        Object carInfoObj = doc.get("carInfo");
        if (carInfoObj instanceof Map) {
            Map<String, Object> carInfo = (Map<String, Object>) carInfoObj;
            info.isCompactCar             = getBool(carInfo, "isCompactCar");
            info.isElectricCar            = getBool(carInfo, "isElectricCar");
            info.isDisabled               = getBool(carInfo, "isDisabled");
            info.isNationalMeritRecipient = getBool(carInfo, "isNationalMeritRecipient");
            info.canUseFemaleParking      = getBool(carInfo, "canUseFemaleParking");
        }

        return info;
    }

    private static boolean getBool(Map<String, Object> map, String key) {
        Object val = map.get(key);
        return val instanceof Boolean && (Boolean) val;
    }
}
