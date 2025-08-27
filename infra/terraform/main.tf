terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ─── Variables ───────────────────────────────────────────────────────────────

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-west1"
}

variable "groq_api_key" {
  description = "Groq API key"
  type        = string
  sensitive   = true
}

variable "gemini_api_key" {
  description = "Gemini API key"
  type        = string
  sensitive   = true
}

# ─── Provider ────────────────────────────────────────────────────────────────

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─── APIs ────────────────────────────────────────────────────────────────────

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# ─── Secrets ─────────────────────────────────────────────────────────────────

resource "google_secret_manager_secret" "groq_key" {
  secret_id = "pulseagent-groq-api-key"
  replication { auto {} }
  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "groq_key" {
  secret      = google_secret_manager_secret.groq_key.id
  secret_data = var.groq_api_key
}

resource "google_secret_manager_secret" "gemini_key" {
  secret_id = "pulseagent-gemini-api-key"
  replication { auto {} }
  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "gemini_key" {
  secret      = google_secret_manager_secret.gemini_key.id
  secret_data = var.gemini_api_key
}

# ─── GCS Bucket (for fixtures & docs) ───────────────────────────────────────

resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-pulseagent-data"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning { enabled = true }

  lifecycle_rule {
    condition { age = 90 }
    action    { type = "Delete" }
  }
}

# ─── Service Account ─────────────────────────────────────────────────────────

resource "google_service_account" "pulseagent" {
  account_id   = "pulseagent-sa"
  display_name = "PulseAgent Service Account"
}

resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.pulseagent.email}"
}

resource "google_storage_bucket_iam_member" "data_access" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.pulseagent.email}"
}

# ─── Cloud Run — API ─────────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "api" {
  name     = "pulseagent-api"
  location = var.region
  depends_on = [google_project_service.run]

  template {
    service_account = google_service_account.pulseagent.email

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = "gcr.io/${var.project_id}/pulseagent-api:latest"

      resources {
        limits = { cpu = "2", memory = "2Gi" }
        cpu_idle = true
      }

      env {
        name = "GROQ_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.groq_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_key.secret_id
            version = "latest"
          }
        }
      }

      ports { container_port = 8000 }
    }
  }
}

# Allow public access to API
resource "google_cloud_run_v2_service_iam_member" "api_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ─── Cloud Run — Dashboard ───────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "dashboard" {
  name     = "pulseagent-dashboard"
  location = var.region
  depends_on = [google_project_service.run]

  template {
    service_account = google_service_account.pulseagent.email

    containers {
      image = "gcr.io/${var.project_id}/pulseagent-dashboard:latest"

      resources {
        limits = { cpu = "1", memory = "1Gi" }
        cpu_idle = true
      }

      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api.uri
      }

      ports { container_port = 8501 }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "dashboard_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.dashboard.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ─── Outputs ─────────────────────────────────────────────────────────────────

output "api_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "PulseAgent API URL"
}

output "dashboard_url" {
  value       = google_cloud_run_v2_service.dashboard.uri
  description = "PulseAgent Dashboard URL"
}

output "data_bucket" {
  value       = google_storage_bucket.data.name
  description = "GCS data bucket name"
}
