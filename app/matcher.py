import os
import cv2
import numpy as np
from app.utils import encode_image_base64, resize_keep_aspect, compute_difference_heatmap

class FaceMatcher:
    def __init__(self, models_dir: str = "models", cosine_threshold: float = 0.363):
        self.models_dir = models_dir
        self.cosine_threshold = cosine_threshold
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
                score_threshold=0.55,
                nms_threshold=0.3,
                top_k=5000
            )
        if os.path.exists(self.recognizer_path):
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.recognizer_path,
                config=""
            )

    def detect_faces(self, image: np.ndarray):
        h, w = image.shape[:2]
        if self.detector is not None:
            self.detector.setInputSize((w, h))
            status, faces = self.detector.detect(image)
            if status and faces is not None and len(faces) > 0:
                return faces

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(rects) > 0:
            synthetic = []
            for r in rects:
                x, y, fw, fh = r
                row = np.zeros(15, dtype=np.float32)
                row[0:4] = [x, y, fw, fh]
                row[4:6] = [x + fw * 0.3, y + fh * 0.35]
                row[6:8] = [x + fw * 0.7, y + fh * 0.35]
                row[8:10] = [x + fw * 0.5, y + fh * 0.55]
                row[10:12] = [x + fw * 0.35, y + fh * 0.75]
                row[12:14] = [x + fw * 0.65, y + fh * 0.75]
                row[14] = 0.90
                synthetic.append(row)
            return np.array(synthetic)
        return None

    def search_person(
        self,
        search_img: np.ndarray,
        reference_vector: list,
        reference_crop_b64: str = None
    ) -> dict:
        img = resize_keep_aspect(search_img, max_dim=1200)
        h, w = img.shape[:2]
        ref_feat = np.array(reference_vector, dtype=np.float32).reshape(1, -1)

        raw_faces = self.detect_faces(img)
        if raw_faces is None or len(raw_faces) == 0:
            return {
                "success": False,
                "match_found": False,
                "total_faces_detected": 0,
                "message": "İkinci görüntüde tespit edilebilir bir insan yüzü bulunamadı.",
                "annotated_image_b64": encode_image_base64(img),
                "candidates": []
            }

        candidates = []
        for idx, face in enumerate(raw_faces):
            box = face[:4].astype(int)
            bx, by, bw, bh = box
            bx = max(0, bx)
            by = max(0, by)
            bw = min(bw, w - bx)
            bh = min(bh, h - by)

            crop = img[by:by+bh, bx:bx+bw]
            if crop.size == 0:
                crop = img

            aligned = None
            if self.recognizer is not None and len(face) >= 14:
                try:
                    aligned = self.recognizer.alignCrop(img, face)
                except Exception:
                    aligned = cv2.resize(crop, (112, 112))
            else:
                aligned = cv2.resize(crop, (112, 112))

            if self.recognizer is not None and aligned is not None:
                feat = self.recognizer.feature(aligned)
                cos_sim = float(self.recognizer.match(ref_feat, feat, cv2.FaceRecognizerSF_FR_COSINE))
                l2_dist = float(self.recognizer.match(ref_feat, feat, cv2.FaceRecognizerSF_FR_NORM_L2))
            else:
                aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if len(aligned.shape) == 3 else aligned
                feat = cv2.resize(aligned_gray, (16, 8)).flatten().astype(np.float32)
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat /= norm
                feat = feat.reshape(1, -1)
                cos_sim = float(np.dot(ref_feat.flatten(), feat.flatten()))
                l2_dist = float(np.linalg.norm(ref_feat - feat))

            sim_normalized = min(100.0, max(0.0, ((cos_sim - 0.15) / (0.80 - 0.15)) * 100.0))
            is_match = cos_sim >= self.cosine_threshold

            candidates.append({
                "index": idx + 1,
                "box": [int(bx), int(by), int(bw), int(bh)],
                "cosine_similarity": round(cos_sim, 4),
                "similarity_pct": round(sim_normalized, 1),
                "l2_distance": round(l2_dist, 4),
                "is_match": is_match,
                "crop_b64": encode_image_base64(aligned if aligned is not None else crop)
            })

        candidates.sort(key=lambda c: c["cosine_similarity"], reverse=True)

        best_match = candidates[0]
        match_found = best_match["is_match"]

        annotated = img.copy()
        for cand in candidates:
            bx, by, bw, bh = cand["box"]
            is_winner = cand["index"] == best_match["index"] and match_found

            if is_winner:
                box_color = (0, 230, 115)
                text = f"HEDEF KISI (%{cand['similarity_pct']:.1f})"
                bg_color = (0, 180, 80)
            else:
                box_color = (80, 100, 140) if not cand["is_match"] else (0, 200, 255)
                text = f"Aday #{cand['index']} (%{cand['similarity_pct']:.1f})"
                bg_color = (40, 50, 70)

            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 3 if is_winner else 2)

            corner_len = min(20, bw // 4, bh // 4)
            if is_winner and corner_len > 5:
                cv2.line(annotated, (bx, by), (bx + corner_len, by), (255, 255, 255), 4)
                cv2.line(annotated, (bx, by), (bx, by + corner_len), (255, 255, 255), 4)
                cv2.line(annotated, (bx + bw, by), (bx + bw - corner_len, by), (255, 255, 255), 4)
                cv2.line(annotated, (bx + bw, by), (bx + bw, by + corner_len), (255, 255, 255), 4)

            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            badge_y1 = max(0, by - th - 12)
            badge_y2 = by
            cv2.rectangle(annotated, (bx, badge_y1), (bx + tw + 16, badge_y2), bg_color, -1)
            cv2.putText(
                annotated,
                text,
                (bx + 8, badge_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        diff_heatmap_b64 = None
        if reference_crop_b64 is not None:
            try:
                from app.utils import decode_image_base64
                ref_crop_img = decode_image_base64(reference_crop_b64)
                cand_crop_img = decode_image_base64(best_match["crop_b64"])
                heatmap_img = compute_difference_heatmap(ref_crop_img, cand_crop_img)
                diff_heatmap_b64 = encode_image_base64(heatmap_img)
            except Exception:
                diff_heatmap_b64 = None

        verdict_title = "Kişi Başarıyla Bulundu!" if match_found else "Hedef Kişi Bu Fotoğrafta Bulunamadı"
        verdict_desc = (
            f"1. Aşamadaki referans yüz vektörü ile taranan {len(candidates)} yüz arasında yapılan kosinüs mesafe analizi sonucunda "
            f"Aday #{best_match['index']} en yüksek skoru elde etti. Benzerlik: %{best_match['similarity_pct']} (Kosinüs: {best_match['cosine_similarity']:.4f}, Eşik: {self.cosine_threshold:.3f})."
        ) if match_found else (
            f"Fotoğrafta {len(candidates)} yüz tespit edildi ancak hiçbir yüz belirlenen biyometrik benzerlik eşiğini ({self.cosine_threshold:.3f}) aşamadı. "
            f"En yakın adayın benzerliği %{best_match['similarity_pct']} seviyesinde kaldı."
        )

        return {
            "success": True,
            "match_found": match_found,
            "total_faces_detected": len(candidates),
            "best_match": best_match,
            "candidates": candidates,
            "annotated_image_b64": encode_image_base64(annotated),
            "diff_heatmap_b64": diff_heatmap_b64,
            "verdict_title": verdict_title,
            "verdict_description": verdict_desc,
            "cosine_threshold": self.cosine_threshold
        }
