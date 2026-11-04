package com.example.myapplication;

import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.GeoPoint;

import java.util.Map;
import java.util.List;

/**
 * Firestore parkingspace 컬렉션의 문서 1건을 표현하는 데이터 클래스
 */
public class ParkingSpot {

    public String id;
    public String name;
    public double latitude;
    public double longitude;
    public String category;   // "public" | "private" | "business"
    public String feeType;    // "paid" | "free" | "conditional_free"
    public Map<String, Object> parkingLots; // 구역별 { total, current }
    public List<Zone> zones = new java.util.ArrayList<>(); // [신설] 도면 zones 목록

    public static class Zone {
        public String id;
        public String name;
        public String bgImgEmpty;
        public float bgScaleX;
        public float bgScaleY;
        public float bgPosX;
        public float bgPosY;
        public float spotOpacity;
        public List<Group> groups = new java.util.ArrayList<>();
        public List<Spot> spots = new java.util.ArrayList<>();
    }

    public static class Group {
        public String id;
        public String name;
        public float x;
        public float y;
        public float rotation;
    }

    public static class Spot {
        public String id;
        public String groupId;
        public float localX;
        public float localY;
        public String type; // "NORMAL" | "COMPACT" | "EV" | "HANDICAP" | "PATRIOT" | "WOMAN"
        public boolean occupied;
    }

    /** Firestore DocumentSnapshot → ParkingSpot 변환 */
    @SuppressWarnings("unchecked")
    public static ParkingSpot fromDocument(DocumentSnapshot doc) {
        ParkingSpot spot = new ParkingSpot();
        spot.id       = doc.getId();
        spot.name     = doc.getString("name");
        spot.category = doc.getString("category");
        spot.feeType  = doc.getString("fee_type");

        // GeoPoint → double 변환
        GeoPoint gp = doc.getGeoPoint("location");
        if (gp != null) {
            spot.latitude  = gp.getLatitude();
            spot.longitude = gp.getLongitude();
        }

        // parking_lots Map 읽기
        Object lotsObj = doc.get("parking_lots");
        if (lotsObj instanceof Map) {
            spot.parkingLots = (Map<String, Object>) lotsObj;
        }

        // [신설] zones 리스트 파싱
        Object zonesObj = doc.get("zones");
        if (zonesObj instanceof List) {
            List<Map<String, Object>> zonesList = (List<Map<String, Object>>) zonesObj;
            for (Map<String, Object> zoneMap : zonesList) {
                Zone zone = new Zone();
                zone.id = (String) zoneMap.get("id");
                zone.name = (String) zoneMap.get("name");
                zone.bgImgEmpty = (String) zoneMap.get("bg_img_empty");
                zone.bgScaleX = getFloat(zoneMap, "bg_scale_x", 1.0f);
                zone.bgScaleY = getFloat(zoneMap, "bg_scale_y", 1.0f);
                zone.bgPosX = getFloat(zoneMap, "bg_pos_x", 0.0f);
                zone.bgPosY = getFloat(zoneMap, "bg_pos_y", 0.0f);
                zone.spotOpacity = getFloat(zoneMap, "spot_opacity", 1.0f);

                // groups 파싱
                Object groupsObj = zoneMap.get("groups");
                if (groupsObj instanceof List) {
                    List<Map<String, Object>> groupsList = (List<Map<String, Object>>) groupsObj;
                    for (Map<String, Object> groupMap : groupsList) {
                        Group group = new Group();
                        group.id = (String) groupMap.get("id");
                        group.name = (String) groupMap.get("name");
                        group.x = getFloat(groupMap, "x", 0.0f);
                        group.y = getFloat(groupMap, "y", 0.0f);
                        group.rotation = getFloat(groupMap, "rotation", 0.0f);
                        zone.groups.add(group);
                    }
                }

                // spots 파싱
                Object spotsObj = zoneMap.get("spots");
                if (spotsObj instanceof List) {
                    List<Map<String, Object>> spotsList = (List<Map<String, Object>>) spotsObj;
                    for (Map<String, Object> spotMap : spotsList) {
                        Spot sp = new Spot();
                        sp.id = (String) spotMap.get("id");
                        sp.groupId = (String) spotMap.get("groupId");
                        sp.localX = getFloat(spotMap, "localX", 0.0f);
                        sp.localY = getFloat(spotMap, "localY", 0.0f);
                        sp.type = (String) spotMap.get("type");
                        Object occObj = spotMap.get("occupied");
                        sp.occupied = occObj instanceof Boolean && (Boolean) occObj;
                        zone.spots.add(sp);
                    }
                }

                spot.zones.add(zone);
            }
        }

        return spot;
    }

    private static float getFloat(Map<String, Object> map, String key, float defaultVal) {
        Object val = map.get(key);
        if (val instanceof Double)  return ((Double) val).floatValue();
        if (val instanceof Long)    return ((Long) val).floatValue();
        if (val instanceof Integer) return ((Integer) val).floatValue();
        if (val instanceof Float)   return (Float) val;
        return defaultVal;
    }

    /** zones 리스트를 Firestore에 저장 가능한 Map의 리스트로 역변환 */
    public List<Map<String, Object>> getZonesMapList() {
        List<Map<String, Object>> list = new java.util.ArrayList<>();
        for (Zone zone : zones) {
            Map<String, Object> zoneMap = new java.util.HashMap<>();
            zoneMap.put("id", zone.id);
            zoneMap.put("name", zone.name);
            zoneMap.put("bg_img_empty", zone.bgImgEmpty);
            zoneMap.put("bg_scale_x", zone.bgScaleX);
            zoneMap.put("bg_scale_y", zone.bgScaleY);
            zoneMap.put("bg_pos_x", zone.bgPosX);
            zoneMap.put("bg_pos_y", zone.bgPosY);
            zoneMap.put("spot_opacity", zone.spotOpacity);

            List<Map<String, Object>> groupsList = new java.util.ArrayList<>();
            if (zone.groups != null) {
                for (Group group : zone.groups) {
                    Map<String, Object> groupMap = new java.util.HashMap<>();
                    groupMap.put("id", group.id);
                    groupMap.put("name", group.name);
                    groupMap.put("x", group.x);
                    groupMap.put("y", group.y);
                    groupMap.put("rotation", group.rotation);
                    groupsList.add(groupMap);
                }
            }
            zoneMap.put("groups", groupsList);

            List<Map<String, Object>> spotsList = new java.util.ArrayList<>();
            if (zone.spots != null) {
                for (Spot spot : zone.spots) {
                    Map<String, Object> spotMap = new java.util.HashMap<>();
                    spotMap.put("id", spot.id);
                    spotMap.put("groupId", spot.groupId);
                    spotMap.put("localX", spot.localX);
                    spotMap.put("localY", spot.localY);
                    spotMap.put("type", spot.type);
                    spotMap.put("occupied", spot.occupied);
                    spotsList.add(spotMap);
                }
            }
            zoneMap.put("spots", spotsList);

            list.add(zoneMap);
        }
        return list;
    }
}
