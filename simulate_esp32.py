import requests

# Your server details
URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "test_car.jpg"
API_KEY = "SmartParking_2026_Secure"

def send_image_like_esp32():
    with open(IMAGE_PATH, 'rb') as f:
        # We send it as a multipart form-data request
        # This automatically handles the 'Boundary' and 'Head/Tail' logic
        files = {'file': (IMAGE_PATH, f, 'image/jpeg')}
        headers = {'Authorization': API_KEY}
        
        print("🚀 Sending image to server...")
        response = requests.post(URL, files=files, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we actually found any plates
            if data['detections']:
                # Get the first detection found
                first_detection = data['detections'][0]
                
                plate = first_detection['plate_number']
                status = first_detection['status']
                owner = first_detection['owner']
                
                print(f"📝 Plate Detected: {plate}")
                print(f"👤 Owner: {owner}")
                print(f"✅ Server Response: {status}")
                
                if status == "Access Granted":
                    print("🔓 LOGIC: Open the Gate!")
                else:
                    print("🔒 LOGIC: Keep Gate Closed.")
            else:
                print("⚠️ No license plates detected in the image.")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    send_image_like_esp32()