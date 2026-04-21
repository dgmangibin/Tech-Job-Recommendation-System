import csv

INPUT_FILE = "AI_Job_Market_Dataset.csv"
OUTPUT_FILE = "AI_Job_Market_Transformed.csv"

SKILL_MAP = {
    "skills_python": "python",
    "skills_sql": "sql",
    "skills_ml": "ML",
    "skills_deep_learning": "deep learning",
    "skills_cloud": "cloud",
}

with open(INPUT_FILE, newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    rows = []
    for row in reader:
        skills = [
            label
            for col, label in SKILL_MAP.items()
            if row.get(col, "0").strip() == "1"
        ]

        rows.append({
            "job_title": row["job_title"],
            "skills": ", ".join(skills) if skills else "",
            "experience": row["experience_level"],
            "salary": row["salary"],
        })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
    fieldnames = ["job_title", "skills", "experience", "salary"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Done! {len(rows)} rows written to {OUTPUT_FILE}")
