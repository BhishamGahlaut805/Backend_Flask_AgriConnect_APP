import cloudinary
import cloudinary.uploader
import cloudinary.api
import io
import logging
from config import Config

logger = logging.getLogger(__name__)

class CloudinaryUploader:
    """Cloudinary uploader for free tier (25GB storage, 25GB bandwidth)"""

    def __init__(self):
        if Config.CLOUDINARY_CLOUD_NAME:
            cloudinary.config(
                cloud_name=Config.CLOUDINARY_CLOUD_NAME,
                api_key=Config.CLOUDINARY_API_KEY,
                api_secret=Config.CLOUDINARY_API_SECRET,
                secure=True
            )
            self.enabled = True
            logger.info("Cloudinary configured successfully")
        else:
            self.enabled = False
            logger.warning("Cloudinary not configured - using GridFS fallback")

    def upload_file(self, file_bytes: bytes, filename: str, folder: str = "uploads") -> dict:
        """Upload file to Cloudinary"""
        if not self.enabled:
            return {"success": False, "error": "Cloudinary not configured"}

        try:
            # Determine resource type
            extension = filename.split('.')[-1].lower()
            if extension in ['mp4', 'avi', 'mov', 'mkv']:
                resource_type = "video"
            elif extension in ['pdf']:
                resource_type = "raw"
            else:
                resource_type = "image"

            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                file_bytes,
                folder=f"agrisupport/{folder}",
                resource_type=resource_type,
                public_id=f"{folder}_{filename.split('.')[0]}_{int(datetime.now().timestamp())}"
            )

            return {
                "success": True,
                "url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "resource_type": resource_type
            }
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {e}")
            return {"success": False, "error": str(e)}

    def delete_file(self, public_id: str, resource_type: str = "image") -> bool:
        """Delete file from Cloudinary"""
        if not self.enabled:
            return False

        try:
            result = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
            return result.get("result") == "ok"
        except Exception as e:
            logger.error(f"Cloudinary delete failed: {e}")
            return False

    def get_url(self, public_id: str, resource_type: str = "image") -> str:
        """Get Cloudinary URL"""
        if not self.enabled:
            return ""

        return cloudinary.CloudinaryImage(public_id).build_url(resource_type=resource_type)


from datetime import datetime