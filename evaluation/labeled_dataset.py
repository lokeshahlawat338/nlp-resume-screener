"""
Labeled evaluation dataset.
Each job description is paired with several resumes, manually ranked by
relevance (1 = best fit for the job, higher numbers = worse fit).

This is the "ground truth" we use to check whether SBERT or TF-IDF
ranks resumes the way a human reviewer would.
"""

LABELED_DATA = [
    {
        "jd_id": "JD_001",
        "job_description": """
            We are looking for a Python Backend Engineer with experience in FastAPI or Django,
            PostgreSQL, Docker, and REST API design. Knowledge of AWS or GCP is a plus.
            The candidate should have strong Git skills and experience with CI/CD pipelines.
        """,
        "resumes": [
            {
                "id": "R1",
                "text": """
                    Software Engineer with 2 years of experience in Python and FastAPI.
                    Built and deployed REST APIs using PostgreSQL and Redis on AWS EC2.
                    Strong Docker and CI/CD experience using GitHub Actions.
                """,
                "relevance_rank": 1
            },
            {
                "id": "R2",
                "text": """
                    Full Stack Developer skilled in Python, Django, and MySQL.
                    Experience with Git, Linux, and basic Docker usage.
                    Built web applications using React and Node.js frontend.
                """,
                "relevance_rank": 2
            },
            {
                "id": "R3",
                "text": """
                    Java Spring Boot engineer with PostgreSQL and Kubernetes experience.
                    Strong backend skills but primary language is Java, not Python.
                    CI/CD using Jenkins and Terraform for infrastructure.
                """,
                "relevance_rank": 3
            },
            {
                "id": "R4",
                "text": """
                    Frontend Developer with expertise in React, TypeScript, and Tailwind CSS.
                    Experience with Figma and UI/UX design principles.
                    No backend or database experience.
                """,
                "relevance_rank": 4
            },
        ]
    },
    {
        "jd_id": "JD_002",
        "job_description": """
            Seeking a Data Scientist / Machine Learning Engineer with strong skills in Python,
            scikit-learn, and TensorFlow or PyTorch. Experience with pandas and NumPy for data
            wrangling is required. Familiarity with NLP techniques is a strong plus.
        """,
        "resumes": [
            {
                "id": "R1",
                "text": """
                    Machine Learning Engineer with 3 years of experience building models in
                    PyTorch and scikit-learn. Strong NLP background, including transformer-based
                    text classification. Heavy use of pandas and NumPy for preprocessing.
                """,
                "relevance_rank": 1
            },
            {
                "id": "R2",
                "text": """
                    Data Analyst proficient in Python, pandas, and SQL. Some exposure to
                    scikit-learn for basic regression models. Mostly focused on dashboards
                    and reporting rather than model building.
                """,
                "relevance_rank": 2
            },
            {
                "id": "R3",
                "text": """
                    Backend Developer with Java and Spring Boot experience. Built data pipelines
                    using Apache Spark. No machine learning or Python experience.
                """,
                "relevance_rank": 4
            },
            {
                "id": "R4",
                "text": """
                    Software Engineer with Python experience in Django web development.
                    Built a small TensorFlow image classifier as a side project.
                    Limited production ML experience.
                """,
                "relevance_rank": 3
            },
        ]
    },
    {
        "jd_id": "JD_003",
        "job_description": """
            We need a DevOps Engineer experienced with Docker, Kubernetes, Terraform, and AWS.
            Strong CI/CD pipeline experience (Jenkins or GitHub Actions) required. Linux
            administration skills are essential.
        """,
        "resumes": [
            {
                "id": "R1",
                "text": """
                    DevOps Engineer with 4 years managing Kubernetes clusters on AWS.
                    Built CI/CD pipelines with Jenkins and Terraform for infrastructure as code.
                    Strong Linux administration background.
                """,
                "relevance_rank": 1
            },
            {
                "id": "R2",
                "text": """
                    Backend Engineer with some Docker experience for local development.
                    Primary focus on Python and Django application code, not infrastructure.
                    Basic AWS S3 and EC2 usage.
                """,
                "relevance_rank": 3
            },
            {
                "id": "R3",
                "text": """
                    Cloud Infrastructure Engineer with Azure and GCP experience.
                    Uses Docker and basic Kubernetes, GitHub Actions for CI/CD.
                    Strong scripting skills in Bash and Python.
                """,
                "relevance_rank": 2
            },
            {
                "id": "R4",
                "text": """
                    Frontend Developer specializing in React and TypeScript.
                    No infrastructure, DevOps, or cloud experience.
                """,
                "relevance_rank": 4
            },
        ]
    },
]