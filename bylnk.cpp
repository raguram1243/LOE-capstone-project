
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>

// Blynk Authentication
char auth[] = "Your_Blynk_Auth_Token";
char ssid[] = "Your_WiFi_SSID";
char pass[] = "Your_WiFi_Password";

// GPS
SoftwareSerial gpsSerial(D5, D6);
TinyGPSPlus gps;

// Ultrasonic
const int trigPin = D1;
const int echoPin = D2;
const int DETECTION_DISTANCE = 15;

// Buzzer for local alert
const int buzzerPin = D3;

// Variables
float latitude, longitude;
unsigned long lastAlertTime = 0;
const unsigned long ALERT_COOLDOWN = 30000;
bool objectDetected = false;
int objectCount = 0;
bool systemActive = true;

// Blynk Virtual Pins
#define VPIN_DISTANCE V0
#define VPIN_OBJECT_STATUS V1
#define VPIN_LATITUDE V2
#define VPIN_LONGITUDE V3
#define VPIN_MAP V4
#define VPIN_OBJECT_COUNT V5
#define VPIN_SYSTEM_STATUS V6
#define VPIN_DETECTION_RANGE V7

void setup() {
  Serial.begin(115200);
 
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
 
  gpsSerial.begin(9600);
  Blynk.begin(auth, ssid, pass);
 
  Serial.println("Object Detection System Started");
  Blynk.virtualWrite(VPIN_SYSTEM_STATUS, "ACTIVE");
}

void loop() {
  Blynk.run();
 
  if (systemActive) {
    readGPS();
    checkObjectDetection();
  }
 
  delay(100);
}

void readGPS() {
  static unsigned long lastGPSTime = 0;
 
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }
 
  if (millis() - lastGPSTime >= 5000) {
    lastGPSTime = millis();
   
    if (gps.location.isUpdated()) {
      latitude = gps.location.lat();
      longitude = gps.location.lng();
     
      Blynk.virtualWrite(VPIN_LATITUDE, latitude, 6);
      Blynk.virtualWrite(VPIN_LONGITUDE, longitude, 6);
      Blynk.virtualWrite(VPIN_MAP, latitude, longitude);
    }
  }
}

float getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
 
  long duration = pulseIn(echoPin, HIGH);
  return duration * 0.034 / 2;
}

void checkObjectDetection() {
  float distance = getDistance();
 
  Blynk.virtualWrite(VPIN_DISTANCE, distance);
 
  if (distance <= DETECTION_DISTANCE && distance > 2) {
    if (!objectDetected) {
      handleObjectDetection(distance);
    }
    // Local buzzer alert
    digitalWrite(buzzerPin, HIGH);
  } else {
    if (objectDetected) {
      objectDetected = false;
      Blynk.virtualWrite(VPIN_OBJECT_STATUS, "CLEAR");
      digitalWrite(buzzerPin, LOW);
    }
  }
}

void handleObjectDetection(float distance) {
  objectDetected = true;
  objectCount++;
 
  Blynk.virtualWrite(VPIN_OBJECT_STATUS, "DETECTED!");
  Blynk.virtualWrite(VPIN_OBJECT_COUNT, objectCount);
 
  sendObjectAlert(distance);
 
  Serial.print("Object detected! Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
}

void sendObjectAlert(float distance) {
  if (millis() - lastAlertTime >= ALERT_COOLDOWN) {
    lastAlertTime = millis();
   
    String message = " Object Detected!\n";
    message += "Distance: " + String(distance, 1) + "cm\n";
    message += "Count: " + String(objectCount);
   
    if (gps.location.isValid()) {
      message += "\n Location: ";
      message += "http://maps.google.com/maps?q=" + String(latitude, 6) + "," + String(longitude, 6);
    }
   
    Blynk.logEvent("object_alert", message);
  }
}

// Blynk Control Handlers
BLYNK_WRITE(VPIN_SYSTEM_STATUS) {
  systemActive = param.asInt();
  digitalWrite(buzzerPin, LOW); // Turn off buzzer when system off
}

BLYNK_WRITE(VPIN_DETECTION_RANGE) {
  int newRange = param.asInt();
  // You can implement dynamic range change here
  Serial.println("Detection range changed to: " + String(newRange));
}

BLYNK_WRITE(V8) { // Reset counter
  if (param.asInt()) {
    objectCount = 0;
    Blynk.virtualWrite(VPIN_OBJECT_COUNT, 0);
    Blynk.virtualWrite(VPIN_OBJECT_STATUS, "RESET");
  }
}

BLYNK_CONNECTED() {
  Blynk.syncAll();
}