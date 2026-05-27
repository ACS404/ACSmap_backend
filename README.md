# ACSmap Backend

> ACSmap is a comprehensive cancer prevention and treatment management platform developed for the American Cancer Society (ACS). This backend server provides REST APIs for user authentication, cancer risk assessment, treatment tracking, medication management, and AI-powered health insights.

## Project Overview

- **Cancer Risk Assessment**: Machine learning-based cancer risk prediction using ACS Cancer Facts & Figures 2026 data
- **Treatment & Medication Tracking**: Track cancer treatments, medications, and daily adherence with personalized schedules
- **User Authentication**: Secure JWT-based authentication with role-based access control (Admin, Educator, Patient)
- **AI-Powered Insights**: Gemini AI integration for personalized cancer risk analysis and medication information
- **Comprehensive Health Profile**: Capture demographic, lifestyle, family history, and medical history data
- **Admin Dashboard**: Administrative tools for managing users, viewing treatment logs, and monitoring patient data
- **HIPAA-Friendly Architecture**: Built with SQLite locally and AWS RDS for production with secure credential management

## Key Features

### 🔬 Cancer Risk Prediction
- **12 Cancer Types**: Lung, colorectal, breast, prostate, melanoma, liver, cervical, stomach, bladder, lymphoma, leukemia, and pancreatic
- **Evidence-Based Modeling**: Uses ACS Cancer Facts & Figures 2026 baseline lifetime risk data
- **Machine Learning**: Ensemble methods (Logistic Regression, Random Forest, Decision Tree) for robust predictions
- **Risk Factors Assessed**:
  - Demographics: Age, sex, race/ethnicity
  - Lifestyle: Smoking status, alcohol consumption, BMI, physical activity, diet quality
  - Medical History: Family history, diabetes, hepatitis, HPV, H. pylori, IBD, radiation exposure, immunosuppression
  - Occupational & Environmental: Chemical exposures, UV exposure
- **Personalized Analysis**: AI-generated narrative explaining risk factors and ACS screening recommendations

### 💊 Treatment & Medication Management
- **Treatment Logging**: Track daily medication adherence with customizable time slots
- **Medication Database**: Search and learn about medications with AI-powered descriptions
- **Flexible Scheduling**: Support for various medication frequencies (daily, weekly, as-needed)
- **Color-Coded Tracking**: Visual tracking with custom colors for different medications
- **Admin Reports**: View all treatments and logs across patients for healthcare providers

### 🔐 User Management
- **Secure Authentication**: JWT tokens with secure HTTP-only cookies
- **Role-Based Access**: Admin, Educator, and Patient roles with appropriate permissions
- **Profile Management**: User profiles with contact information and preferences
- **Password Reset**: Secure password reset capabilities for administrators

### 🤖 AI Integration
- **Gemini AI Chat**: Ask questions about cancer prevention, risk factors, and screening guidelines
- **Medication Information**: AI-generated summaries of medication uses and side effects
- **Personalized Risk Narratives**: 3-paragraph personalized cancer risk explanations based on individual profiles
- **ACS-Aligned Responses**: All AI responses grounded in ACS Cancer Facts & Figures 2026 data

## Quick Start

> Prerequisites: Python 3.9 or later (macOS, WSL Ubuntu, or Ubuntu)

### Clone and Setup

```bash
mkdir -p ~/acsmap
cd ~/acsmap
git clone https://github.com/ACS404/ACSmap_backend.git
cd ACSmap_backend
```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file in the project root:

```shell
# Server Configuration
FLASK_PORT=8009
SECRET_KEY=your-secret-key-here

# JWT Token Configuration
JWT_TOKEN_NAME=jwt_python_flask

# Admin user defaults
ADMIN_USER='Admin User'
ADMIN_UID='admin'
ADMIN_PASSWORD='admin123'
DEFAULT_PASSWORD='password123'

# Default test user
USER_NAME='Test Patient'
USER_UID='testpatient'
USER_PASSWORD='testpass123'

# Teacher/Educator defaults
TEACHER_USER='Health Educator'
TEACHER_UID='educator'
TEACHER_PASSWORD='educator123'

# AI Services
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_SERVER=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent

GROQ_API_KEY=your_groq_api_key_here
GROQ_SERVER=https://api.groq.com/openai/v1/chat/completions

# Database Configuration
IS_PRODUCTION=false
DB_USERNAME='admin'
DB_PASSWORD='your_db_password_here'
```

### Initialize Database

```bash
python scripts/db_init.py
```

### Run the Server

```bash
# Development with debug mode
python main.py

# Server will start at http://localhost:8009
```

## VSCode Setup

1. **Open in VSCode:**
   ```bash
   code .
   ```

2. **Configure Python Interpreter:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Search: `Python: Select Interpreter`
   - Choose `./venv/bin/python`

3. **Install Extensions:**
   - Python
   - Pylance
   - SQLite3 Editor

4. **View Database:**
   - Navigate to `instance/volumes/user_management.db`
   - Open with SQLite3 Editor to inspect tables

5. **Start Debugging:**
   - Open `main.py`
   - Press F5 or click the Play button
   - Click localhost link in terminal to launch

## API Endpoints

### Authentication & User Management
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|----------------|
| POST | `/api/authenticate` | Login user and receive JWT token | No |
| GET | `/api/id` | Get current logged-in user profile | Yes |
| POST | `/api/user` | Create new user account (sign up) | No |
| GET | `/api/users` | List all users (admin only) | Yes (Admin) |
| PUT | `/api/user/<id>` | Update user profile | Yes |
| DELETE | `/api/user/<id>` | Delete user account | Yes (Admin) |

### Cancer Risk Assessment
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|----------------|
| GET | `/api/cancer-risk/predict` | Get overall cancer risk prediction | Yes |
| POST | `/api/cancer-risk/predict` | Calculate cancer risk from patient data | Yes |
| GET | `/api/cancer-risk/predict-types` | Get per-cancer-type risk breakdown | Yes |
| GET | `/api/cancer-risk/factors` | Get list of identified risk factors | Yes |
| GET | `/api/cancer-risk/feature-importance` | Get feature importance for risk model | Yes |
| POST | `/api/cancer/risk-analysis` | Generate personalized AI risk analysis | Yes |

### Treatment & Medication Management
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|----------------|
| GET | `/api/treatments` | Get all active treatments for user | Yes |
| POST | `/api/treatments` | Create new treatment/medication | Yes |
| PUT | `/api/treatments` | Update existing treatment | Yes |
| DELETE | `/api/treatments` | Soft-delete treatment (mark inactive) | Yes |
| GET | `/api/treatment/log` | Get treatment logs for specific date | Yes |
| POST | `/api/treatment/log` | Log medication taken (adherence tracking) | Yes |
| GET | `/api/medication/info?name=X` | Get AI-generated medication description | Yes |

### Admin & Reporting
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|----------------|
| GET | `/api/admin/treatments` | View all patient treatments | Yes (Admin) |
| GET | `/api/admin/treatment/logs` | View all patient treatment logs | Yes (Admin) |
| GET | `/api/admin/treatment/notes` | View all patient treatment notes | Yes (Admin) |
| DELETE | `/api/admin/treatment/notes` | Delete treatment notes | Yes (Admin) |

### AI Chatbot
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|----------------|
| POST | `/api/acs-chat` | Chat with ACS cancer risk assistant | Yes |

### Social Features
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|----------------|
| GET | `/api/post/all` | Get all social posts | Yes |
| POST | `/api/post` | Create new post | Yes |
| GET | `/api/microblog` | Get microblog posts with filters | Yes |
| POST | `/api/microblog` | Create microblog post | Yes |

## Database Models

### User
```python
- uid: String (unique identifier)
- name: String
- email: String
- password: String (hashed with salt)
- role: String (Admin, Educator, or Patient)
- created_at: DateTime
- profile_picture: String (optional)
```

### Treatment
```python
- id: Integer (primary key)
- user_id: Integer (foreign key)
- medication_name: String (required)
- dosage: String (e.g., "500 mg")
- frequency: String (daily, weekly, etc.)
- times: JSON list of times (e.g., ["08:00", "20:00"])
- color: String (hex color for UI)
- notes: Text (custom user notes)
- ai_description: Text (cached from Gemini)
- start_date: Date
- end_date: Date (nullable)
- active: Boolean (soft delete flag)
- created_at: DateTime
```

### TreatmentLog
```python
- id: Integer (primary key)
- treatment_id: Integer (foreign key)
- user_id: Integer (foreign key)
- log_date: Date (when medication was tracked)
- time_slot: String (e.g., "08:00" or "anytime")
- taken: Boolean (was medication taken)
- taken_at: DateTime (timestamp when marked as taken)
```

### CancerRiskProfile
```python
- Demographics: age, sex, race/ethnicity
- Lifestyle: smoking_status, pack_years, bmi, alcohol, physical_activity, diet
- Medical History: family_history, diabetes, hepatitis, hpv, h_pylori, ibd
- Environmental: occupational_exposure, uv_exposure, radiation_history
- Genetic: genetic_mutation flags
```

## Project Structure

```
ACSmap_backend/
├── main.py                      # Application entry point
├── __init__.py                  # Flask app initialization
├── api/                         # REST API blueprints
│   ├── authenticate.py          # Authentication endpoints
│   ├── treatment.py             # Treatment/medication endpoints
│   ├── cancer_risk.py           # Cancer risk prediction endpoints
│   ├── acs_chat_api.py          # Gemini AI chat endpoint
│   ├── treatment_notes_api.py   # Treatment notes endpoints
│   └── ...                      # Other API modules
├── model/                       # SQLAlchemy database models
│   ├── user.py                  # User model
│   ├── treatment.py             # Treatment & TreatmentLog models
│   ├── cancer_risk.py           # Cancer risk ML model
│   ├── treatment_notes.py       # Treatment notes model
│   └── ...                      # Other data models
├── scripts/                     # Utility scripts
│   ├── db_init.py              # Initialize/reset database
│   ├── db_migrate-prod2sqlite.py # Pull production DB to local
│   └── db_restore-sqlite2prod.py # Push local DB to production
├── templates/                   # Jinja2 HTML templates (admin UI)
├── static/                      # Static files (CSS, JS, images)
├── instance/volumes/            # Runtime SQLite database storage
│   └── user_management.db      # Main application database
├── requirements.txt             # Python package dependencies
└── .env                        # Environment variables (create this)
```

## Usage Examples

### Login and Get JWT Token

```bash
curl -X POST http://localhost:8009/api/authenticate \
  -H "Content-Type: application/json" \
  -d '{
    "uid": "testpatient",
    "password": "testpass123"
  }'

# Response includes JWT token to use in subsequent requests
```

### Check Cancer Risk

```bash
curl -X POST http://localhost:8009/api/cancer-risk/predict \
  -H "Content-Type: application/json" \
  -H "Cookie: jwt_python_flask=YOUR_JWT_TOKEN" \
  -d '{
    "age": 55,
    "sex": "male",
    "race": "white",
    "smoking_status": "former",
    "pack_years": 15,
    "bmi_category": "overweight",
    "alcohol_consumption": "moderate",
    "physical_activity": "moderate",
    "diet_quality": "average",
    "family_history": true,
    "diabetes": false,
    "hepatitis": false,
    "hpv": false,
    "h_pylori": false,
    "ibd": false,
    "radiation_history": false,
    "immunosuppression": false,
    "precancerous_lesions": false,
    "occupational_exposure": false,
    "uv_exposure": false
  }'
```

### Add a Medication to Track

```bash
curl -X POST http://localhost:8009/api/treatments \
  -H "Content-Type: application/json" \
  -H "Cookie: jwt_python_flask=YOUR_JWT_TOKEN" \
  -d '{
    "medication_name": "Tamoxifen",
    "dosage": "20 mg",
    "frequency": "daily",
    "times": ["08:00", "20:00"],
    "color": "#FF6B6B",
    "notes": "Take with food",
    "start_date": "2026-05-27"
  }'
```

### Log Medication Adherence

```bash
curl -X POST http://localhost:8009/api/treatment/log \
  -H "Content-Type: application/json" \
  -H "Cookie: jwt_python_flask=YOUR_JWT_TOKEN" \
  -d '{
    "treatment_id": 1,
    "date": "2026-05-27",
    "time_slot": "08:00",
    "taken": true
  }'
```

### Get Medication Information

```bash
curl "http://localhost:8009/api/medication/info?name=Tamoxifen" \
  -H "Cookie: jwt_python_flask=YOUR_JWT_TOKEN"
```

### Chat with ACS Cancer Assistant

```bash
curl -X POST http://localhost:8009/api/acs-chat \
  -H "Content-Type: application/json" \
  -H "Cookie: jwt_python_flask=YOUR_JWT_TOKEN" \
  -d '{
    "message": "What lifestyle changes can reduce my lung cancer risk?"
  }'
```

## Development Workflow

### Testing Locally

1. **Initialize clean database:**
   ```bash
   python scripts/db_init.py
   ```

2. **Make code changes and test:**
   - Edit files in `model/`, `api/`, or `main.py`
   - Restart the server (Flask debug mode auto-reloads)
   - Test endpoints with curl or Postman

3. **Test cancer risk prediction:**
   ```bash
   python -m model.cancer_risk
   ```

### Production Deployment

1. **On production server:**
   ```bash
   # Backup current database
   cp instance/volumes/user_management.db \
      instance/volumes/backups/backup_$(date +%Y%m%d_%H%M%S).db
   
   # Update code
   git pull
   
   # Apply any schema changes
   python scripts/db_init.py
   
   # Restart Flask service
   systemctl restart flask-app  # or your service name
   ```

## Technology Stack

- **Backend**: Python 3.9+ with Flask microframework
- **Database**: SQLite (development), AWS RDS PostgreSQL (production)
- **Authentication**: JWT (JSON Web Tokens) with secure cookies
- **ORM**: SQLAlchemy for database abstraction
- **Machine Learning**: scikit-learn for cancer risk prediction
- **AI Services**: Google Gemini for natural language analysis
- **Deployment**: Docker, AWS, Ubuntu/Linux with Nginx or Apache
- **Security**: Bcrypt for password hashing, CORS support

## Cancer Risk Model Details

### Training Data
- 10,000 synthetic patient profiles based on US population demographics
- Features engineered from ACS Cancer Facts & Figures 2026 data
- Risk scores calibrated to match ACS baseline lifetime incidence rates

### Models Used
- **Logistic Regression**: For binary high/low risk classification
- **Random Forest**: Ensemble method with 100 trees for robust predictions
- **Decision Tree**: Feature importance extraction

### Output
- Low/high risk probability
- Risk category (low or high)
- Model confidence score
- Overall relative risk multiplier
- Per-cancer-type risks (12 cancer types)
- Key contributing factors with explanations
- Actionable ACS screening recommendations

## Security Considerations

- **Password Storage**: Bcrypt with salt (not stored in plaintext)
- **JWT Tokens**: HS256 algorithm, 24-hour expiration (configurable)
- **HTTP-Only Cookies**: Prevents JavaScript access to tokens
- **CORS**: Configurable for frontend domain
- **Admin Functions**: Role-based access control (requires Admin role)
- **Data Privacy**: Support for user data export and deletion

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support & Documentation

For more information:
- [American Cancer Society](https://www.cancer.org) - Cancer facts and screening guidelines
- [ACS Cancer Facts & Figures 2026](https://www.cancer.org/research/cancer-facts-statistics/all-cancer-facts-figures.html)
- Reach out to the development team for technical questions

## Disclaimer

This application is for educational and informational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.
