#!/usr/bin/env python3

import face_recognition
import pickle
from pathlib import Path
import argparse

# Veri klasörleri
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Path objelerini oluştur
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Training klasöründeki resimleri yükleyip face encoding'lerini oluşturur
    """
    names = []
    encodings = []
    
    # Training klasöründeki her alt klasörü gez
    for person_dir in Path("training").iterdir():
        if person_dir.is_dir():
            person_name = person_dir.name
            print(f"🔍 {person_name} için resimler işleniyor...")
            
            # Her kişinin klasöründeki resimleri işle
            for image_path in person_dir.glob("*"):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"   📸 İşleniyor: {image_path.name}")
                    
                    # Resmi yükle
                    image = face_recognition.load_image_file(image_path)
                    
                    # Yüz encoding'ini çıkar
                    face_encodings = face_recognition.face_encodings(image, model=model)
                    
                    if face_encodings:
                        # İlk yüzü al (tek yüz olduğunu varsayıyoruz)
                        encoding = face_encodings[0]
                        encodings.append(encoding)
                        names.append(person_name)
                        print(f"   ✅ {person_name} için encoding oluşturuldu")
                    else:
                        print(f"   ❌ {image_path.name} dosyasında yüz bulunamadı!")

    # Encodings'leri kaydet
    name_encodings = {"names": names, "encodings": encodings}
    
    with open(encodings_location, "wb") as f:
        pickle.dump(name_encodings, f)
    
    print(f"\n🎉 Encoding'ler kaydedildi: {encodings_location}")
    print(f"📊 Toplam {len(names)} yüz encoding'i oluşturuldu")
    
    # İstatistikler
    unique_names = set(names)
    for name in unique_names:
        count = names.count(name)
        print(f"   - {name}: {count} encoding")

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Verilen resimdeki yüzleri tanır
    """
    # Encoding'leri yükle
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

    # Test resmini yükle
    input_image = face_recognition.load_image_file(image_location)

    # Test resmindeki yüzleri bul
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    print(f"🔍 Test resminde {len(input_face_encodings)} yüz bulundu")

    # Her yüzü tanımaya çalış
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        # Bu yüzün bilinen yüzlerle karşılaştır
        matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        
        name = "Bilinmeyen"
        if True in matches:
            # İlk eşleşmeyi al
            first_match_index = matches.index(True)
            name = loaded_encodings["names"][first_match_index]

        print(f"   👤 Bulunan kişi: {name}")

def validate(model: str = "hog"):
    """
    Validation klasöründeki resimleri test eder
    """
    print("🧪 Validation testi başlıyor...")
    
    validation_dir = Path("validation")
    if not any(validation_dir.iterdir()):
        print("❌ Validation klasörü boş!")
        return
    
    for image_path in validation_dir.glob("*"):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"\n📸 Test ediliyor: {image_path.name}")
            recognize_faces(str(image_path), model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument(
        "--train", action="store_true", help="Training verilerini işle ve model oluştur"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validation verilerini test et"
    )
    parser.add_argument(
        "--test", "-t", help="Belirtilen resmi test et"
    )
    parser.add_argument(
        "--model", 
        default="hog", 
        choices=["hog", "cnn"], 
        help="Kullanılacak model (hog: hızlı, cnn: daha doğru)"
    )

    args = parser.parse_args()

    if args.train:
        print("🚀 Training başlıyor...")
        encode_known_faces(model=args.model)
    
    if args.validate:
        validate(model=args.model)
    
    if args.test:
        print(f"🔍 Test ediliyor: {args.test}")
        recognize_faces(args.test, model=args.model)
    
    if not (args.train or args.validate or args.test):
        print("📖 Kullanım:")
        print("  python detector.py --train        # Model eğit")
        print("  python detector.py --validate     # Validation test et")
        print("  python detector.py --test resim.jpg  # Tek resmi test et")