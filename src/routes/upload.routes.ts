import { Router, Request, Response, NextFunction } from "express";
import multer from "multer";
import { processReport } from "../controllers/upload.controller";

const router = Router();
const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024,
    files: 5,
  },
  fileFilter: (req, file, cb) => {
    const allowed = [
      "application/pdf",
      "image/jpeg",
      "image/png",
      "image/webp",
      "text/plain",
      "audio/mpeg",
      "audio/wav",
    ];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type detected: ${file.mimetype}`));
    }
  },
});

router.post(
  "/",
  (req: Request, res: Response, next: NextFunction) => {
    

    upload.any()(req, res, (err: any) => {
      if (err) {
        console.error("🚨 [MULTER MIDDLEWARE CRASH]:", err.message);
        return res.status(400).json({ success: false, error: err.message });
      }
 
    
      next();
    });
  },
  processReport,
);

export default router;
