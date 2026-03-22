from flask import Flask, render_template, request
import re
from difflib import get_close_matches
from groq import Groq
from dotenv import load_dotenv
import os

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# =====================
# TEXT UTILS
# =====================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def fuzzy_match(word, keywords):
    return bool(get_close_matches(word, keywords, cutoff=0.8))

# =====================
# KEYWORDS
# =====================

PROMPT_KEYWORDS = {

    "leadership": [
        "lead", "led", "leadership", "managed", "organized",
        "initiated", "founded", "president", "captain",
        "representative", "rep", "council", "role", "position",
        "serve", "elected", "volunteer"
    ],
    "community": [
        "community", "team", "others", "collaborate",
        "students", "school", "peers", "grade", "class",
        "represent", "voice", "support", "together"
    ],

    "interest": [
        "interest", "passion", "curious", "motivated",
        "excited", "drawn"
    ],
    "skills": [
        "skills", "experience", "proficient",
        "knowledge", "trained", "developed"
    ],

    "ideas": [
        "built", "created", "designed", "proposed", "developed",
        "launched", "started", "project", "initiative", "organized"
    ],

    "impact": [
        "impact", "improved", "increased",
        "reduced", "results", "difference", "helped"
    ],
    "growth": [
        "challenge", "learned", "mistake", "improved"
    ],
   
    "identity": [
        "background", "culture", "belief", "identity"
    ]
}

EXPERIENCE_KEYWORDS = {
    "leadership": [
        "president", "captain", "leader", "organized", "led", "lead",
        "managed", "directed", "coordinated", "founded", "created",
        "started", "ran", "chaired", "headed", "supervised", "mentored",
        "guided", "delegated", "oversaw", "spearheaded", "initiated"
    ],
    "interest": [
        "interested", "passionate", "curious", "love", "loved",
        "enjoy", "enjoyed", "fascinated", "drawn", "motivated",
        "inspired", "explored", "pursued", "dedicated", "committed",
        "self-taught", "independently", "hobby", "outside", "personal"
    ],
    "skills": [
        "coding", "python", "writing", "programming", "design",
        "research", "analysis", "built", "developed", "created",
        "taught", "tutored", "trained", "studied", "learned",
        "practiced", "applied", "used", "worked", "proficient",
        "experienced", "skilled", "knowledge", "technical", "data",
        "math", "science", "engineering", "statistics", "excel",
        "presented", "published", "computed", "modeled", "tested"
    ],
    "impact": [
        "improved", "helped", "increased", "changed", "raised",
        "reduced", "grew", "expanded", "saved", "earned", "won",
        "achieved", "accomplished", "resulted", "contributed",
        "supported", "assisted", "benefited", "served", "impacted",
        "transformed", "solved", "fixed", "built", "launched",
        "students", "community", "others", "people", "members",
        "school", "team", "organization", "club", "raised", "funds"
    ],
    "experiences": [
        "volunteer", "intern", "member", "tutor", "job", "work",
        "worked", "position", "role", "club", "team", "organization",
        "program", "class", "course", "project", "competition",
        "award", "scholarship", "research", "lab", "hospital",
        "nonprofit", "summer", "camp", "conference", "workshop"
    ]
}

# =====================
# CORE LOGIC
# =====================

def classify_prompt(prompt):
    words = normalize(prompt)
    scores = {cat: 0 for cat in PROMPT_KEYWORDS}

    for word in words:
        for cat, kws in PROMPT_KEYWORDS.items():
            if word in kws:
                scores[cat] += 1
            elif fuzzy_match(word, kws):
                scores[cat] += 0.5

    return [cat for cat, score in scores.items() if score > 0]


def relevance_score(exp, categories):
    score = 0
    exp_words = normalize(exp)

    for cat in categories:
        keywords = EXPERIENCE_KEYWORDS.get(cat, [])
        for word in exp_words:
            if word in keywords:
                score += 2
            elif fuzzy_match(word, keywords):
                score += 1

    score += 1
    return score


def match_experiences(experiences, categories):
    scored = [(exp, relevance_score(exp, categories)) for exp in experiences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [exp for exp, score in scored]


# =====================
# AI FUNCTIONS
# =====================

def explain_prompt(prompt, categories, extra_context=""):
    system_prompt = """You are an application coach for high school students. They are applying to competitive clubs and programs, not schools or universities. Do not use first person.
Analyze the given application prompt and return exactly this format with no extra text:

Meaning: [one sentence — what the reviewer is really looking for]
Mistake: [one sentence — the most common mistake students make on this type of prompt]
Strategy: [exactly 3 short numbered tips, each under 20 words, specific to this prompt]

Be specific to the actual prompt. Talk directly to the student using 'you'. No fluff."""

    user_message = f"""Application prompt: "{prompt}"
What this prompt is testing: {', '.join(categories)}"""

    if extra_context:
        user_message += f"\nExtra context from the student: {extra_context}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300
        )

        text = response.choices[0].message.content.strip()

        meaning, mistake, strategy = "", "", ""
        current = None

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("meaning"):
                meaning = line.split(":", 1)[-1].strip()
                current = "meaning"
            elif line.lower().startswith("mistake"):
                mistake = line.split(":", 1)[-1].strip()
                current = "mistake"
            elif line.lower().startswith("strategy"):
                strategy = line.split(":", 1)[-1].strip()
                current = "strategy"
            else:
                if current == "meaning":
                    meaning += " " + line
                elif current == "mistake":
                    mistake += " " + line
                elif current == "strategy":
                    strategy += "\n" + line

        return {"Meaning": meaning, "Mistake": mistake, "Strategy": strategy}

    except Exception as e:
        print("GROQ ERROR explain_prompt:", e)
        return {
            "Meaning": "The reviewer wants to understand what makes you a strong candidate.",
            "Mistake": "Being too generic — write something only you could write.",
            "Strategy": "1. Be specific\n2. Use your real experiences\n3. Connect to the role"
        }


def generate_outline(category, top_experience, prompt, extra_context=""):
    system_prompt = """You are an application coach for high school students applying to competitive clubs or programs.
 
A strong application response must:
- Reveal something genuine about who the student is as a person
- Avoid generic statements that any student could write
- Highlight a spike — one thing they are deeply invested in or uniquely experienced in
- Connect to their broader ambitions or goals
- Feel like it has a consistent theme or narrative thread

Your job is to give 4 specific things this student should make sure to include in their response. Do NOT invent or assume specific details, moments, or stories the student did not provide. Only reference what the student actually told you.
These are NOT structural steps — they are specific content suggestions tailored to their experience and prompt.
Each point tells them WHAT to actually say, not how to format their answer.
Each point is ONE sentence under 20 words.
Do NOT start with any intro phrase. Start directly with '1.'
Do NOT give generic advice like 'be specific' or 'show passion'.
No bold text. Talk directly to the student using 'you'."""

    user_message = f"""Prompt the student is answering: "{prompt}"
Their most relevant experience: {top_experience}
What the prompt is testing: {category}"""

    if extra_context:
        user_message += f"\nExtra context from the student: {extra_context}"

    user_message += "\n\nWhat 4 specific things should this student make sure to include in their response?"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=350
        )
        text = response.choices[0].message.content.strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines

    except Exception as e:
        print("GROQ ERROR generate_outline:", e)
        return [
            f"1. Mention a specific moment from {top_experience} that reveals who you are.",
            "2. Show your spike — the one thing you are more invested in than anything else.",
            "3. Connect your experience to your broader goals and ambitions.",
            "4. End with something only you could write — your unique angle."
        ]


def reflection_questions(category):
    questions = {
        "interest": [
            "What moment sparked this interest?",
            "Why this club or program over others?"
        ],
        "leadership": [
            "What decision did YOU make?",
            "What was challenging?"
        ],
        "skills": [
            "Where did you develop this skill?",
            "How will it help you contribute?"
        ],
        "impact": [
            "Who benefited from your actions?",
            "Can you quantify the result?"
        ]
    }
    return questions.get(category, [])


# =====================
# FLASK ROUTE
# =====================

@app.route("/", methods=["GET"])
@app.route("/analyze", methods=["POST"])
def index():
    results = None

    if request.method == "POST":
        prompt = request.form["prompt"]
        raw = request.form.get("experiences", "")
        experiences = [e.strip() for e in raw.split(",") if e.strip()]
        extra_context = request.form.get("context", "").strip()

        categories = classify_prompt(prompt)
        matched = match_experiences(experiences, categories)
        top_experience = matched[0] if matched else "your experience"

        prompt_explanation = explain_prompt(prompt, categories[:1], extra_context)

        results = {
            "categories": categories,
            "experiences": matched,
            "prompt_explanation": prompt_explanation,
            "outlines": {
                cat: generate_outline(cat, top_experience, prompt, extra_context)
                for cat in categories[:1]
            },
            "questions": {
                cat: reflection_questions(cat)
                for cat in categories
            },
        }

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)