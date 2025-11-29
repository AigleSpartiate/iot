#include <SPI.h>
#include <Ethernet.h>
#include <PubSubClient.h>
#include <DHT.h>

// ====== DHT CONFIG ======
#define LIGHT_SENSOR_PIN A1
#define SOUND_SENSOR_PIN A2
#define AIR_QUALITY_PIN A3
#define DHT_PIN 8       // <-- use the digital pin your Grove sensor is on
#define DHT_TYPE DHT11  // Grove basic T&H = DHT11;
DHT dht(DHT_PIN, DHT_TYPE);

// ====== NETWORK CONFIG ======
byte mac[] = { 0xA8, 0x61, 0x0A, 0xAF, 0x15, 0x6C };
IPAddress ip(192, 168, 10, 2);           // Arduino static IP
IPAddress mqtt_server(192, 168, 10, 1);  // Raspberry Pi static IP

// ====== ROOM INFO ======
#define ROOM_ID 1
#define ROOM_NAME "Room 1"

EthernetClient ethClient;
PubSubClient client(ethClient);

const char* mqtt_topic = "test/room1/temperature";
// measurement counter
unsigned long measurementCount = 0;
unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL = 5000;  // every 5 seconds

void reconnectMQTT() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("arduino_temp_test")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(9600);

  dht.begin();  // start Grove T&H sensor

  // init pins
  pinMode(LIGHT_SENSOR_PIN, INPUT);
  pinMode(SOUND_SENSOR_PIN, INPUT);
  pinMode(AIR_QUALITY_PIN, INPUT);

  Ethernet.begin(mac, ip);
  delay(1000);
  Serial.print("Arduino IP: ");
  Serial.println(Ethernet.localIP());

  client.setServer(mqtt_server, 1883);
}

void loop() {
  if (!client.connected()) {
    reconnectMQTT();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastSend >= SEND_INTERVAL) {
    lastSend = now;
    sendData();
  }
}

void sendData() {

  // increment measurement id
  measurementCount++;

  // Read from Grove DHT sensor
  float temperature = dht.readTemperature();  // °C
  float humidity = dht.readHumidity();        // %
  int light = readLightSensor();
  int sound = readSoundSensor();
  int airQualityValue = readAirQuality();
  String airQualityLevel = getAirQualityLevel(airQualityValue);

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT read failed");
    return;  // don't publish bad data
  }

  /* JSON format: simple format to send via MQTT
  {
  "room_id": 1,
  "name": "Room 1",
  "measurement_id": 46,
  "uptime": 230130,
  "temperature": 20.90,
  "humidity": 51.90,
  "light": 248,
  "sound": 985,
  "airQuality": 61,
  "airQuality-Level": "good"
  }
  */

  // JSON with temperature (and humidity if you want)
  String payload = "{";
  // room (id)
  payload += "\"room_id\":";
  payload += ROOM_ID;

  // room name
  payload += ",\"name\":\"";
  payload += ROOM_NAME;
  payload += "\"";

  // measurement id
  payload += ",\"measurement_id\":";
  payload += measurementCount;

  // uptime in ms
  payload += ",\"uptime\":";
  payload += millis();
  payload += ",\"temperature\":";
  payload += String(temperature, 2);
  payload += ",\"humidity\":";
  payload += String(humidity, 2);
  payload += ",\"light\":";
  payload += String(light);
  payload += ",\"sound\":";
  payload += String(sound);
  payload += ",\"airQuality\":";
  payload += String(airQualityValue);
  payload += ",\"airQuality-Level\":\"";
  payload += airQualityLevel;
  payload += "\"";
  payload += "}";

  Serial.print("Publishing: ");
  Serial.println(payload);

  if (client.publish(mqtt_topic, payload.c_str())) {
    Serial.println("MQTT publish: OK");
  } else {
    Serial.println("MQTT publish: FAILED");
  }
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
  if (value < 50) {
    return "excellent";  // fresh air
  } else if (value < 200) {
    return "good";  // low pollution
  } else if (value < 400) {
    return "moderate";  // high pollution
  } else if (value < 600) {
    return "poor";  // hazardous
  } else {
    return "hazardous";  // very hazardous
  }
}