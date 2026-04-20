import os
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

training_data = [
    # User Action
    ("User User1234 logged in.", "User Action"),
    ("User User5678 logged out.", "User Action"),
    ("User User9999 logged in.", "User Action"),
    ("Account with ID 1234 created by AdminUser.", "User Action"),
    ("Account with ID 5678 created by User99.", "User Action"),
    ("User admin logged in from IP 10.0.0.1.", "User Action"),
    ("New user account registered for email user@example.com.", "User Action"),
    ("User profile updated for user ID 3421.", "User Action"),
    ("Password changed for user ID 7890.", "User Action"),
    ("User User4321 logged out after session timeout.", "User Action"),

    # System Notification
    ("Backup completed successfully.", "System Notification"),
    ("Backup started at 2024-01-01 00:00:00.", "System Notification"),
    ("Backup ended at 2024-01-01 01:00:00.", "System Notification"),
    ("System updated to version 3.5.1.", "System Notification"),
    ("Disk cleanup completed successfully.", "System Notification"),
    ("System reboot initiated by user 00412.", "System Notification"),
    ("File data_1234.csv uploaded successfully by user User123.", "System Notification"),
    ("Scheduled maintenance completed without errors.", "System Notification"),
    ("Database backup finished. Size: 2.3GB.", "System Notification"),
    ("Server health check passed. All services running normally.", "System Notification"),
    ("Cache cleared successfully on server Node-7.", "System Notification"),
    ("SSL certificate renewed successfully for domain example.com.", "System Notification"),

    # Security Alert
    ("IP 192.168.1.1 blocked due to potential attack.", "Security Alert"),
    ("Admin access escalation detected for user 9429.", "Security Alert"),
    ("Multiple login failures detected for user 54231.", "Security Alert"),
    ("Unauthorized access attempt detected from IP 10.10.10.10.", "Security Alert"),
    ("Suspicious activity flagged for account ID 8823.", "Security Alert"),
    ("Brute force attack detected on port 22.", "Security Alert"),
    ("User account locked after 5 failed login attempts.", "Security Alert"),
    ("Firewall rule triggered: blocked outbound traffic to 45.33.32.156.", "Security Alert"),
    ("SQL injection attempt detected in request from IP 172.16.0.5.", "Security Alert"),
    ("DDoS attack mitigated. Traffic normalized after 10 minutes.", "Security Alert"),

    # HTTP Log
    ("GET /v2/3454/servers/detail HTTP/1.1 RCODE 404 len: 1583 time: 0.1878400", "HTTP Log"),
    ("GET /v2/servers/detail HTTP/1.1 RCODE 200 len: 1583 time: 0.1878400", "HTTP Log"),
    ("POST /api/v1/login HTTP/1.1 RCODE 200 len: 342 time: 0.0521", "HTTP Log"),
    ("DELETE /api/v1/users/123 HTTP/1.1 RCODE 403 len: 89 time: 0.0312", "HTTP Log"),
    ("PUT /api/v1/products/456 HTTP/1.1 RCODE 200 len: 512 time: 0.0893", "HTTP Log"),
    ("alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error", "HTTP Log"),
    ("GET /health HTTP/1.1 RCODE 200 len: 15 time: 0.002", "HTTP Log"),
    ("POST /v1/messages HTTP/1.1 RCODE 201 len: 1024 time: 0.2341", "HTTP Log"),

    # Unclassified
    ("Hey bro, chill ya!", "Unclassified"),
    ("Random text that doesn't match anything.", "Unclassified"),
    ("Testing 1 2 3.", "Unclassified"),
    ("Lorem ipsum dolor sit amet.", "Unclassified"),
]

texts  = [t for t, _ in training_data]
labels = [l for _, l in training_data]

print("Loading sentence transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
embeddings = embedder.encode(texts, show_progress_bar=True)

print("Training classifier...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(embeddings, labels)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/log_classifier.joblib")
print("✅ Saved to models/log_classifier.joblib")
