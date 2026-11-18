# Save this as test_setup.py and run it: python3 test_setup.py
from osgeo import gdal
from google.cloud import storage

print(f"GDAL version: {gdal.VersionInfo()}")

# Try to list your organization's bucket (Replace with your bucket name)
try:
    client = storage.Client()
    buckets = list(client.list_buckets(max_results=1))
    print("GCP Access: SUCCESS")
except Exception as e:
    print(f"GCP Access: FAILED - {e}")
