import os
import random
import uuid
import json
import datetime
from faker import Faker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageDraw

DOCUMENT_TYPES = [
    "university_degree",
    "transcript",
    "professional_license",
    "employment_letter",
    "diploma",
    "certificate"
]

LOCALES = ['en_US', 'ar_SA', 'es_ES', 'fr_FR', 'hi_IN']

fakers = {loc: Faker(loc) for loc in LOCALES}

def draw_seal(c, x, y, radius=40, text="OFFICIAL SEAL"):
    c.saveState()
    c.setStrokeColor(colors.darkred)
    c.setFillColor(colors.whitesmoke)
    c.setLineWidth(3)
    c.circle(x, y, radius, fill=1)
    c.setFillColor(colors.darkred)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(x, y, text)
    c.restoreState()

def draw_border(c, width, height, margin=30):
    c.saveState()
    c.setStrokeColor(colors.darkblue)
    c.setLineWidth(4)
    c.rect(margin, margin, width - 2*margin, height - 2*margin)
    c.restoreState()

def gen_university_degree(c, fake, w, h):
    name = fake.name()
    institution = fake.company() + " University"
    date = fake.date_this_decade().strftime('%Y-%m-%d')
    c.setFont("Helvetica-Bold", 35)
    c.drawCentredString(w/2, h - 150, "University Degree")
    c.setFont("Helvetica", 18)
    c.drawCentredString(w/2, h - 250, "This certifies that")
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w/2, h - 320, name)
    c.setFont("Helvetica", 18)
    c.drawCentredString(w/2, h - 380, "has attained a degree from")
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(w/2, h - 440, institution)
    c.setFont("Helvetica", 14)
    c.drawString(100, 150, f"Issued Date: {date}")
    draw_seal(c, w - 150, 150, 50, "UNIVERSITY SEAL")
    return {"name": name, "institution": institution, "date": date}

def gen_transcript(c, fake, w, h):
    name = fake.name()
    institution = fake.company()
    date = fake.date_this_decade().strftime('%Y-%m-%d')
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, h - 100, "ACADEMIC TRANSCRIPT")
    c.setFont("Helvetica", 12)
    c.drawString(50, h - 140, f"Student: {name}")
    c.drawString(50, h - 160, f"Institution: {institution}")
    c.drawString(50, h - 180, f"Date: {date}")
    
    y = h - 240
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Subject")
    c.drawString(400, y, "Grade")
    y -= 20
    c.setFont("Helvetica", 12)
    subjects = [fake.job() for _ in range(5)]
    grades = ['A', 'B', 'C', 'A+', 'B+']
    for _ in range(random.randint(4, 7)):
        c.drawString(50, y, fake.job()[:30])
        c.drawString(400, y, random.choice(grades))
        y -= 20
        
    draw_seal(c, w - 100, h - 100, 30, "REGISTRAR")
    return {"name": name, "institution": institution, "date": date}

def gen_professional_license(c, fake, w, h):
    name = fake.name()
    profession = fake.job()
    date = fake.date_between(start_date='today', end_date='+5y').strftime('%Y-%m-%d')
    draw_border(c, w, h, 20)
    
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w/2, h - 150, "Professional License")
    
    c.setFont("Helvetica", 16)
    c.drawCentredString(w/2, h - 230, "This license is granted to")
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(w/2, h - 280, name)
    c.setFont("Helvetica", 16)
    c.drawCentredString(w/2, h - 340, "to practice as a certified")
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(w/2, h - 390, profession)
    
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(w/2, h - 480, f"Valid Until: {date}")
    return {"name": name, "profession": profession, "valid_until": date}

def gen_employment_letter(c, fake, w, h):
    name = fake.name()
    company = fake.company()
    date = fake.date_this_year().strftime('%Y-%m-%d')
    c.setFont("Helvetica", 12)
    c.drawString(50, h - 100, company)
    c.drawString(50, h - 120, fake.address().replace('\n', ', '))
    c.drawString(50, h - 160, f"Date: {date}")
    c.drawString(50, h - 200, f"To Whom It May Concern,")
    
    body = (f"This letter confirms that {name} is currently employed "
            f"at {company}. They have been a valued member of our team.")
    c.drawString(50, h - 250, body)
    
    c.drawString(50, h - 350, "Sincerely,")
    c.drawString(50, h - 390, fake.name())
    c.drawString(50, h - 410, "Human Resources")
    return {"name": name, "company": company, "date": date}

def gen_diploma(c, fake, w, h):
    name = fake.name()
    school = fake.company() + " Academy"
    date = fake.date_this_decade().strftime('%Y-%m-%d')
    draw_border(c, w, h, 40)
    
    c.setFont("Times-BoldItalic", 45)
    c.drawCentredString(w/2, h - 180, "Diploma of Excellence")
    c.setFont("Times-Roman", 20)
    c.drawCentredString(w/2, h - 280, "Presented to")
    c.setFont("Helvetica-Bold", 35)
    c.drawCentredString(w/2, h - 350, name)
    c.setFont("Times-Roman", 20)
    c.drawCentredString(w/2, h - 430, f"by the faculty of {school}")
    c.setFont("Helvetica", 14)
    c.drawString(w/2 - 50, h - 550, f"Date: {date}")
    
    c.saveState()
    c.setStrokeColor(colors.gold)
    c.setLineWidth(5)
    c.circle(100, 100, 30)
    c.restoreState()
    
    return {"name": name, "institution": school, "date": date}

def gen_certificate(c, fake, w, h):
    name = fake.name()
    date = fake.date_this_year().strftime('%Y-%m-%d')
    reason = fake.bs().title()
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w/2, h - 150, "Certificate of Achievement")
    c.setFont("Helvetica", 18)
    c.drawCentredString(w/2, h - 250, "Awarded to")
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w/2, h - 320, name)
    c.setFont("Helvetica", 18)
    c.drawCentredString(w/2, h - 380, "For outstanding performance in")
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, h - 440, reason)
    c.setFont("Helvetica", 14)
    c.drawString(100, 150, f"Date: {date}")
    draw_seal(c, w - 100, 150, 40, "CERTIFIED")
    return {"name": name, "reason": reason, "date": date}

GENERATORS = {
    "university_degree": gen_university_degree,
    "transcript": gen_transcript,
    "professional_license": gen_professional_license,
    "employment_letter": gen_employment_letter,
    "diploma": gen_diploma,
    "certificate": gen_certificate
}

def apply_noise(pdf_path):
    """
    Renders PDF to image, applies PIL noise (blur, rot, stains) to make it realistic 
    and saves back as PDF.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    
    # Render at 200 DPI
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    
    # Apply Noise
    # 1. Blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    # 2. Rotation
    if random.random() < 0.5:
        angle = random.uniform(-3.0, 3.0)
        img = img.rotate(angle, expand=True, fillcolor="white")
        
    # 3. Coffee stains
    if random.random() < 0.7:
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, img.width)
            y = random.randint(0, img.height)
            r = random.randint(20, 80)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(139, 69, 19, round(255 * 0.15)))  # Transparent brown
    
    # Save back to PDF over the existing file
    img.save(pdf_path, "PDF", resolution=200.0)


def generate_synthetic_data(num_per_type=167):
    base_dir = "data/synthetic_docs"
    os.makedirs(base_dir, exist_ok=True)
    
    metadata = {}
    
    for doc_type in DOCUMENT_TYPES:
        type_dir = os.path.join(base_dir, doc_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for _ in range(num_per_type):
            doc_id = str(uuid.uuid4())
            locale = random.choice(LOCALES)
            fake = fakers[locale]
            
            pdf_path = os.path.join(type_dir, f"doc_{doc_id}.pdf")
            c = canvas.Canvas(pdf_path, pagesize=letter)
            w, h = letter
            
            generator_func = GENERATORS[doc_type]
            fields = generator_func(c, fake, w, h)
            c.save()
            
            # 30% chance to add noise
            is_noisy = random.random() < 0.3
            if is_noisy:
                apply_noise(pdf_path)
                
            metadata[f"doc_{doc_id}.pdf"] = {
                "document_id": doc_id,
                "document_type": doc_type,
                "language": locale,
                "has_noise": is_noisy,
                "extracted_fields": fields
            }
            
    # Save ground truth metadata
    with open(os.path.join(base_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully generated {len(metadata)} documents in {base_dir}")


if __name__ == "__main__":
    import time
    start = time.time()
    generate_synthetic_data(167)
    print(f"Took {time.time() - start:.2f} seconds.")
