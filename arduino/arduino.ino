/*
 * Outputs JSON formatted sensor data via Serial
 */

#include <DHT.h>

// ============== PIN DEFINITIONS ==============
#define LIGHT_SENSOR_PIN A1
#define SOUND_SENSOR_PIN A2
#define AIR_QUALITY_PIN A3
#define DHT_PIN 8

// ============== SENSOR CONFIGURATION ==============
#define DHT_TYPE DHT11
#define ROOM_NAME "Room_1"
#define ROOM_ID 1

// ============== TIMING ==============
#define MEASUREMENT_INTERVAL 60000  // 1 minute in ms

// ============== SENSOR OBJECTS ==============
DHT dht(DHT_PIN, DHT_TYPE);

// ============== VARIABLES ==============
unsigned long lastMeasurement = 0;
unsigned long measurementCount = 0;

// calibrated during warmup
int airQualityBaseline = 0;

void setup() {
    Serial.begin(9600);
    while (!Serial) {
        ; // wait for serial port to connect
    }
    
    // init DHT sensor
    dht.begin();
    
    // init pins
    pinMode(LIGHT_SENSOR_PIN, INPUT);
    pinMode(SOUND_SENSOR_PIN, INPUT);
    pinMode(AIR_QUALITY_PIN, INPUT);
    
    Serial.println("{\"status\": \"warming_up\", \"message\": \"Air quality sensor warming up...\"}");
    delay(20000);  // 20 second warmup for air quality sensor
    
    // calibrate air quality baseline
    airQualityBaseline = analogRead(AIR_QUALITY_PIN);
    
    Serial.println("{\"status\": \"ready\", \"message\": \"Sensor system initialized\"}");
    
    // take first measurement immediately
    takeMeasurement();
}

void loop() {
    unsigned long currentMillis = millis();
    
    if (currentMillis - lastMeasurement >= MEASUREMENT_INTERVAL) {
        lastMeasurement = currentMillis;
        takeMeasurement();
    }
}

void takeMeasurement() {
    measurementCount++;
    
    // read all sensors
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();
    int lightValue = readLightSensor();
    int soundValue = readSoundSensor();
    int airQualityValue = readAirQuality();
    String airQualityLevel = getAirQualityLevel(airQualityValue);
    
    // check for DHT read errors
    bool dhtError = isnan(temperature) || isnan(humidity);
    
    // output JSON
    Serial.print("{");
    Serial.print("\"room_id\":");
    Serial.print(ROOM_ID);
    Serial.print(",\"room_name\":\"");
    Serial.print(ROOM_NAME);
    Serial.print("\",\"measurement_id\":");
    Serial.print(measurementCount);
    Serial.print(",\"uptime_ms\":");
    Serial.print(millis());
    Serial.print(",\"sensors\":{");
    
    // Temp
    Serial.print("\"temperature\":{");
    if (!dhtError) {
        Serial.print("\"value\":");
        Serial.print(temperature, 2);
        Serial.print(",\"unit\":\"celsius\"");
    } else {
        Serial.print("\"error\":\"read_failed\"");
    }
    Serial.print("},");
    
    // Humidity
    Serial.print("\"humidity\":{");
    if (!dhtError) {
        Serial.print("\"value\":");
        Serial.print(humidity, 2);
        Serial.print(",\"unit\":\"percent\"");
    } else {
        Serial.print("\"error\":\"read_failed\"");
    }
    Serial.print("},");
    
    // Light
    Serial.print("\"light\":{");
    Serial.print("\"raw\":");
    Serial.print(lightValue);
    Serial.print(",\"percent\":");
    Serial.print(map(lightValue, 0, 1023, 0, 100));
    Serial.print("},");
    
    // Sound
    Serial.print("\"sound\":{");
    Serial.print("\"raw\":");
    Serial.print(soundValue);
    Serial.print(",\"percent\":");
    Serial.print(map(soundValue, 0, 1023, 0, 100));
    Serial.print("},");
    
    // Air Quality
    Serial.print("\"air_quality\":{");
    Serial.print("\"raw\":");
    Serial.print(airQualityValue);
    Serial.print(",\"level\":\"");
    Serial.print(airQualityLevel);
    Serial.print("\"}");
    
    Serial.println("}}");
}

int readLightSensor() {
    return analogRead(LIGHT_SENSOR_PIN);
}

int readSoundSensor() {
    // multiple samples for more accurate sound reading
    long total = 0;
    for (int i = 0; i < 32; i++) {
        total += analogRead(SOUND_SENSOR_PIN);
        delayMicroseconds(100);
    }
    return total / 32;
}

int readAirQuality() {
    // average of multiple readings
    long total = 0;
    for (int i = 0; i < 10; i++) {
        total += analogRead(AIR_QUALITY_PIN);
        delay(10);
    }
    return total / 10;
}

String getAirQualityLevel(int value) {
    // Grove Air Quality Sensor thresholds TODO: check
    if (value < 100) {
        return "excellent";
    } else if (value < 200) {
        return "good";
    } else if (value < 400) {
        return "moderate";
    } else if (value < 700) {
        return "poor";
    } else {
        return "hazardous";
    }
}