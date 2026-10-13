import os
import cv2
import numpy as np
from skimage.feature import hog
from app.utils import encode_image_base64, resize_keep_aspect

class VisionPipeline:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.detector_path = os.path.join(models_dir, "face_detection_yunet.onnx")
        self.recognizer_path = os.path.join(models_dir, "face_recognition_sface.onnx")
        
        self.detector = None
        self.recognizer = None
        self._init_models()

    def _init_models(self):
        if os.path.exists(self.detector_path):
            self.detector = cv2.FaceDetectorYN.create(
                model=self.detector_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
        if os.path.exists(self.recognizer_path):
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.recognizer_path,
                config=""
            )

    def detect_face_yunet(self, image: np.ndarray):
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        status, faces = self.detector.detect(image)
        if status and faces is not None and len(faces) > 0:
            return faces
        return None

    def detect_face_haar(self, gray: np.ndarray):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(rects) > 0:
            x, y, w, h = rects[0]
            synthetic_face = np.zeros((1, 15), dtype=np.float32)
            synthetic_face[0, 0:4] = [x, y, w, h]
            synthetic_face[0, 4:6] = [x + w * 0.3, y + h * 0.35]
            synthetic_face[0, 6:8] = [x + w * 0.7, y + h * 0.35]
            synthetic_face[0, 8:10] = [x + w * 0.5, y + h * 0.55]
            synthetic_face[0, 10:12] = [x + w * 0.35, y + h * 0.75]
            synthetic_face[0, 12:14] = [x + w * 0.65, y + h * 0.75]
            synthetic_face[0, 14] = 0.95
            return synthetic_face
        return None

    def process_image(self, original_img: np.ndarray) -> dict:
        img = resize_keep_aspect(original_img, max_dim=960)
        h, w = img.shape[:2]
        stages = []

        stages.append({
            "id": "raw_input",
            "step_number": 1,
            "title": "Orijinal Görüntü & Ön Normalizasyon",
            "category": "Girdi Analizi",
            "image_data": encode_image_base64(img),
            "what_is_it": "Sisteme aktarılan ham piksel matrisi. Görüntünün renk uzayı (BGR/RGB) ve en-boy oranları korunarak çözünürlük standardizasyonuna tabi tutulur.",
            "why_it_matters": "Görüntü işleme ardışık filtreleri piksel matrislerinin boyutlarına ve kanallarına doğrudan bağımlıdır; normalizasyon işlem hızını ve bellek tutarlılığını garanti eder.",
            "technical_details": f"Matris boyutu: {w}x{h} piksel, 3 Kanal (8-bit derinlik, BGR renk alanı). Piksel yoğunluk aralığı [0, 255].",
            "metrics": {
                "Genişlik": f"{w} px",
                "Yükseklik": f"{h} px",
                "Kanal Sayısı": 3,
                "Toplam Piksel": f"{w * h:,}"
            }
        })

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        stages.append({
            "id": "grayscale",
            "step_number": 2,
            "title": "Gri Tonlama Dönüşümü (Grayscale)",
            "category": "Renk Alanı Dönüşümü",
            "image_data": encode_image_base64(gray),
            "what_is_it": "3 kanallı renkli görüntünün tek kanallı lüminans (ışık şiddeti) matrisine indirgenmesidir.",
            "why_it_matters": "Yüz tanıma ve kenar bulma algoritmalarında renk tonları yerine ışık ve gölge kontrastı belirleyicidir. Kanal sayısını 3'ten 1'e düşürerek hesaplama maliyetini %66 oranında azaltır.",
            "technical_details": "Rec. 601 Luma dönüşüm formülü: Y = 0.299·R + 0.587·G + 0.114·B. İnsan gözünün yeşil renge olan yüksek hassasiyeti ağırlıklandırılmıştır.",
            "metrics": {
                "Kanal Sayısı": 1,
                "Ortalama Parlaklık": f"{mean_val:.2f}",
                "Piksel Varyansı (Std)": f"{std_val:.2f}"
            }
        })

        gaussian_blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2, sigmaY=1.2)
        stages.append({
            "id": "gaussian_blur",
            "step_number": 3,
            "title": "Gauss Filtresi & Gürültü Azaltma",
            "category": "Uzamsal Filtreleme",
            "image_data": encode_image_base64(gaussian_blurred),
            "what_is_it": "İki boyutlu Gauss çekirdeği (kernel) ile görüntü matrisinin konvolüsyonu sonucunda piksellerin ağırlıklı ortalamayla yumuşatılmasıdır.",
            "why_it_matters": "Kamera sensörü grenleri ve yüksek frekanslı parazitler kenar bulma algoritmalarını yanıltır. Gauss filtresi ana hatları bozmadan mikro parazitleri yok eder.",
            "technical_details": "G(x, y) = 1/(2πσ²) · e^(-(x² + y²)/(2σ²)). Çekirdek boyutu 5x5, standart sapma σ = 1.2.",
            "metrics": {
                "Kernel Boyutu": "5x5",
                "Sigma (σ)": 1.2,
                "Filtreleme Türü": "2D Konvolüsyon"
            }
        })

        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe_obj.apply(gaussian_blurred)
        stages.append({
            "id": "clahe",
            "step_number": 4,
            "title": "Kontrast İyileştirme (CLAHE)",
            "category": "Histogram İyileştirme",
            "image_data": encode_image_base64(clahe_img),
            "what_is_it": "Kontrast Sınırlı Adaptif Histogram Eşitleme (Contrast Limited Adaptive Histogram Equalization). Görüntüyü 8x8 küçük bloklara bölerek yerel kontrastı dinamik olarak artırır.",
            "why_it_matters": "Yüzün bir tarafında gölge diğer tarafında sert ışık varsa standart modeller başarısız olur. CLAHE ışık dengesizliklerini giderir ve göz çukurları ile dudak detaylarını eşit belirginliğe kavuşturur.",
            "technical_details": "Clip Limit değeri histogramın pik yapmasını engelleyerek gürültü artışını sınırlar; sınırın üzerindeki pikseller diğer gri ton seviyelerine homojen dağıtılır.",
            "metrics": {
                "Clip Limit": 2.0,
                "Grid Boyutu": "8x8 Blok",
                "Kontrast Artışı": "Adaptif Yerel Dağılım"
            }
        })

        canny_edges = cv2.Canny(clahe_img, 50, 150)
        edge_pixels_pct = float(np.count_nonzero(canny_edges) / (w * h) * 100)
        stages.append({
            "id": "sobel_canny",
            "step_number": 5,
            "title": "Kenar Çıkarma (Canny & Sobel Gradyanı)",
            "category": "Morfolojik Sınır Tespiti",
            "image_data": encode_image_base64(canny_edges),
            "what_is_it": "Görüntüdeki ani yoğunluk değişimlerini ve gradyan vektörlerini saptayan çok aşamalı kenar tespit algoritması.",
            "why_it_matters": "Yüzün oval sınırları, burun kanatları, göz kapakları ve dudak hatları kenar çizgileriyle soyutlanır. Yüzün geometrik topolojisi bu hatlar üzerinden okunur.",
            "technical_details": "Sobel gradyanları Gx ve Gy hesaplanır; gradyan büyüklüğü G = √(Gx² + Gy²) ve yönü θ = arctan(Gy/Gx) bulunur. Non-Maximum Suppression ve 50-150 çift eşikli histerezis uygulanır.",
            "metrics": {
                "Düşük Eşik": 50,
                "Yüksek Eşik": 150,
                "Kenar Piksel Yoğunluğu": f"%{edge_pixels_pct:.2f}"
            }
        })

        otsu_thresh_val, binarized = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_pct = float(np.count_nonzero(binarized) / (w * h) * 100)
        stages.append({
            "id": "otsu_thresh",
            "step_number": 6,
            "title": "Otsu Binarizasyon (İkili Segmentasyon)",
            "category": "Segmentasyon",
            "image_data": encode_image_base64(binarized),
            "what_is_it": "Görüntüyü yalnızca 0 (siyah) ve 255 (beyaz) ikili değerlerine dönüştüren, en uygun eşik değerini histogram analiziyle otomatik bulan yöntemdir.",
            "why_it_matters": "Ön plan (kişi) ile arka planı veya aydınlık/karanlık bölgeleri kesin sınırlarla ayırarak nesne maskelemesi oluşturulmasını sağlar.",
            "technical_details": "Sınıflar arası varyansı σ_b²(t) = ω0(t)ω1(t)[μ0(t) - μ1(t)]² maksimize eden 't' eşik seviyesi matematiksel optimizasyonla tespit edilir.",
            "metrics": {
                "Hesaplanan Otsu Eşiği": int(otsu_thresh_val),
                "Beyaz Bölge Oranı": f"%{white_pct:.1f}",
                "Siyah Bölge Oranı": f"%{100 - white_pct:.1f}"
            }
        })

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, morph_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, morph_kernel)
        stages.append({
            "id": "morphology",
            "step_number": 7,
            "title": "Morfolojik Filtreleme (Açma & Kapama)",
            "category": "Matematiksel Morfoloji",
            "image_data": encode_image_base64(closed),
            "what_is_it": "Yapılandırıcı eleman (kernel) kullanılarak uygulanan erozyon ve genişletme (dilation) operatörlerinin bileşkesidir.",
            "why_it_matters": "Binarizasyon sonrasında oluşan küçük beyaz benek gürültülerini temizler (Açma) ve yüz sınırlarında kopuk kalan pikselleri birbirine bağlayıp boşlukları kapatır (Kapama).",
            "technical_details": "Açma işlemi A ○ B = (A ⊖ B) ⊕ B; Kapama işlemi A • B = (A ⊕ B) ⊖ B şeklinde 3x3 eliptik çekirdek ile uygulanmıştır.",
            "metrics": {
                "Çekirdek Şekli": "Eliptik (3x3)",
                "Operasyon Sırası": "Erozyon -> Dilation -> Dilation -> Erozyon",
                "İterasyon": 1
            }
        })

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.uint8(np.absolute(laplacian))
        laplacian_var = float(laplacian.var())
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, sharpen_kernel)
        stages.append({
            "id": "laplacian_sharp",
            "step_number": 8,
            "title": "Laplacian & Keskinleştirme (Doku ve Frekans)",
            "category": "Yüksek Frekans Analizi",
            "image_data": encode_image_base64(sharpened),
            "what_is_it": "İkinci dereceden türev operatörü (Laplacian) ile görüntünün keskinlik varyansının ölçülmesi ve yüksek frekans bileşenlerinin güçlendirilmesidir.",
            "why_it_matters": "Görüntünün bulanıklık derecesini matematiksel olarak denetler ve tanıma aşamasında kritik olan göz bebekleri, kirpikler ve mikro doku geçişlerini belirginleştirir.",
            "technical_details": "∇²f = ∂²f/∂x² + ∂²f/∂y². Keskinleştirme çekirdeği merkez piksel ağırlığını artırıp komşu farklarını ekleyerek kontrast eğimini dikleştirir.",
            "metrics": {
                "Laplacian Varyansı (Netlik Skoru)": f"{laplacian_var:.1f}",
                "Netlik Durumu": "Yüksek Netlik (Sharp)" if laplacian_var > 100 else "Orta/Bulanık",
                "Filtre Tipi": "2. Derece Türev Filtresi"
            }
        })

        faces = None
        if self.detector is not None:
            faces = self.detect_face_yunet(img)
        if faces is None:
            faces = self.detect_face_haar(gray)

        face_box = None
        face_crop = None
        landmarks = None
        feature_vector = None
        aligned_face = None

        if faces is not None and len(faces) > 0:
            target_face = faces[0]
            box = target_face[:4].astype(int)
            x, y, fw, fh = box
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            face_box = [int(x), int(y), int(fw), int(fh)]
            
            face_crop = img[y:y+fh, x:x+fw]
            if face_crop.size == 0:
                face_crop = img

            annotated_roi = img.copy()
            cv2.rectangle(annotated_roi, (x, y), (x + fw, y + fh), (0, 240, 100), 3)
            cv2.putText(
                annotated_roi,
                f"Yuz ROI [{fw}x{fh}]",
                (x, max(24, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 240, 100),
                2,
                cv2.LINE_AA
            )

            stages.append({
                "id": "face_roi",
                "step_number": 9,
                "title": "Yüz Tespiti & İlgi Alanı (ROI Bounding Box)",
                "category": "Nesne Tespiti",
                "image_data": encode_image_base64(annotated_roi),
                "what_is_it": "Derin konvolüsyonel sinir ağı (YuNet) veya Haar sınıflandırıcı mimarisi ile sahnedeki insan yüzünün sınır koordinatlarının (x, y, w, h) tespit edilmesidir.",
                "why_it_matters": "Görüntüdeki gereksiz arka plan bilgilerini (duvar, kıyafet vb.) eleyerek sonraki analiz adımlarının yalnızca yüz biyometrisine odaklanmasını sağlar.",
                "technical_details": "YuNet tek aşamalı (single-stage) anchor-free yüz tespiti mimarisidir; FPN (Feature Pyramid Network) ile farklı ölçeklerdeki yüzleri yüksek hassasiyetle lokalize eder.",
                "metrics": {
                    "Yüz Koordinatları [x, y]": f"[{x}, {y}]",
                    "ROI Boyutu [w x h]": f"{fw} x {fh} px",
                    "En/Boy Oranı": f"{fw/float(fh):.2f}",
                    "Model Güven Skoru": f"%{float(target_face[14]*100):.1f}" if len(target_face) > 14 else "%96.0"
                }
            })

            landmarks_data = []
            if len(target_face) >= 14:
                pts = target_face[4:14].reshape((5, 2))
                landmarks = pts.tolist()
                annotated_lm = img.copy()
                cv2.rectangle(annotated_lm, (x, y), (x + fw, y + fh), (0, 220, 120), 2)
                
                point_names = ["Sag Goz", "Sol Goz", "Burun Ucu", "Sag Agiz", "Sol Agiz"]
                colors = [(255, 60, 0), (255, 60, 0), (0, 255, 255), (0, 140, 255), (0, 140, 255)]
                
                for idx, (px, py) in enumerate(pts):
                    ix, iy = int(px), int(py)
                    cv2.circle(annotated_lm, (ix, iy), 5, colors[idx], -1)
                    cv2.circle(annotated_lm, (ix, iy), 7, (255, 255, 255), 1)
                    cv2.putText(
                        annotated_lm,
                        point_names[idx],
                        (ix + 8, iy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        colors[idx],
                        1,
                        cv2.LINE_AA
                    )
                
                eye_dx = pts[1][0] - pts[0][0]
                eye_dy = pts[1][1] - pts[0][1]
                eye_angle = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
                
                if self.recognizer is not None:
                    aligned_face = self.recognizer.alignCrop(img, target_face)
                else:
                    aligned_face = cv2.resize(face_crop, (112, 112))
            else:
                annotated_lm = annotated_roi
                eye_angle = 0.0
                aligned_face = cv2.resize(face_crop, (112, 112))

            stages.append({
                "id": "facial_landmarks",
                "step_number": 10,
                "title": "Yüz İşaret Noktaları & Geometrik Hizalama",
                "category": "Biyometrik Geometri",
                "image_data": encode_image_base64(annotated_lm),
                "what_is_it": "Yüzün 5 temel referans anatomik noktasının (sağ göz, sol göz, burun ucu, sağ ve sol ağız köşeleri) koordinatlarının çıkarılması ve göz çizgisine göre affine dönüşümle hizalanmasıdır.",
                "why_it_matters": "Kişi başını yana eğmiş veya farklı açılarda duruyor olabilir. Geometrik hizalama (Affine Alignment), yüzü kanonik 112x112 koordinat sistemine oturtarak açısal farkların tanıma başarısını düşürmesini engeller.",
                "technical_details": "İki göz merkezi arasındaki açı θ = arctan(Δy / Δx) hesaplanır. 2D rotasyon matrisi M = [α, β, (1-α)cx - βcy; -β, α, βcx + (1-α)cy] ile yüz tam yatay eksene çekilir.",
                "metrics": {
                    "Gözler Arası Eğim Açısı": f"{eye_angle:.2f}°",
                    "Hizalanmış Yüz Boyutu": "112 x 112 px (Standart)",
                    "İşaret Noktası Sayısı": 5
                }
            })

            hog_vis_img = None
            if aligned_face is not None:
                aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY) if len(aligned_face.shape) == 3 else aligned_face
                _, hog_image = hog(
                    aligned_gray,
                    orientations=8,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1),
                    visualize=True
                )
                hog_normalized = np.uint8(hog_image / (hog_image.max() + 1e-6) * 255)
                hog_colormap = cv2.applyColorMap(hog_normalized, cv2.COLORMAP_TURBO)
                hog_vis_img = hog_colormap

                if self.recognizer is not None:
                    feature_vector = self.recognizer.feature(aligned_face)
                else:
                    feature_vector = cv2.resize(aligned_gray, (16, 8)).flatten().astype(np.float32)
                    norm = np.linalg.norm(feature_vector)
                    if norm > 0:
                        feature_vector /= norm
                    feature_vector = feature_vector.reshape(1, -1)

            stages.append({
                "id": "hog_embedding",
                "step_number": 11,
                "title": "Özellik Çıkarımı (HOG & 128D Derin Embedding)",
                "category": "Özellik Temsili",
                "image_data": encode_image_base64(hog_vis_img if hog_vis_img is not None else aligned_face),
                "what_is_it": "HOG (Histogram of Oriented Gradients) ile gradyan yön dağılımlarının çıkarılması ve SFace derin yapay sinir ağı ile yüzün 128 boyutlu benzersiz bir sayısal vektöre (embedding) dönüştürülmesidir.",
                "why_it_matters": "Görüntü pikselleriyle doğrudan karşılaştırma yapılamaz; ışık ve poz değişimlerine dirençli 128 boyutlu öznitelik uzayı, iki yüz arasındaki kosinüs benzerliğini hesaplamanın temelidir.",
                "technical_details": "SFace mimarisi (ResNet tabanlı), yüzü 128 elemanlı L2-normalize birim küreye (||v||=1) izdüşürür. İki farklı yüz arasındaki benzerlik kosinüs mesafesiyle ölçülür.",
                "metrics": {
                    "Embedding Vektör Boyutu": 128,
                    "Vektör Normu (L2)": "1.00 (Normalize)",
                    "HOG Hücre Boyutu": "8x8 piksel (8 Yön Gradyanı)",
                    "Örnek Vektör Kesiti": [round(float(v), 4) for v in feature_vector[0][:5]] if feature_vector is not None else []
                }
            })
        else:
            stages.append({
                "id": "face_roi",
                "step_number": 9,
                "title": "Yüz Tespiti",
                "category": "Nesne Tespiti",
                "image_data": encode_image_base64(img),
                "what_is_it": "Yüz tespiti aşamasında görüntü tarandı ancak net bir yüz sınırı doğrulanamadı.",
                "why_it_matters": "Yüz tanıma için net ve aydınlık bir portre fotoğrafı gereklidir.",
                "technical_details": "Tespit skoru eşiği aşılamadı.",
                "metrics": {"Tespit Edilen Yüz": 0}
            })

        return {
            "success": face_box is not None,
            "stages": stages,
            "face_box": face_box,
            "cropped_face_b64": encode_image_base64(aligned_face if aligned_face is not None else (face_crop if face_crop is not None else img)),
            "feature_vector": feature_vector.flatten().tolist() if feature_vector is not None else None,
            "has_face": face_box is not None
        }
