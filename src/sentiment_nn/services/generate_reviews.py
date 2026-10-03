"""Generate a synthetic G2/Capterra-style review dataset for sentiment practice."""
import random
import csv
from datetime import date, timedelta

random.seed(7)

PRODUCTS = {
    "CRM": ["Pipely", "DealNest", "ClientOrbit"],
    "Project Management": ["TaskHarbor", "Sprintly Pro", "PlanBoard"],
    "Help Desk": ["TicketTide", "SupportLoop", "HelpHive"],
    "Email Marketing": ["MailMosaic", "SendGarden", "InboxRocket"],
    "Accounting": ["LedgerLeaf", "BookBridge", "CountWise"],
    "HR Software": ["PeoplePort", "StaffSpring", "HireCanvas"],
    "Analytics": ["MetricMint", "InsightForge", "ChartCove"],
}

FEATURE = {
    "CRM": ["pipeline view", "contact management", "deal tracking", "email sync", "sales reports"],
    "Project Management": ["kanban boards", "Gantt chart", "task dependencies", "time tracking", "team workload view"],
    "Help Desk": ["ticket routing", "shared inbox", "knowledge base", "SLA tracking", "live chat widget"],
    "Email Marketing": ["drag-and-drop editor", "automation workflows", "A/B testing", "list segmentation", "campaign reports"],
    "Accounting": ["invoicing", "bank reconciliation", "expense tracking", "tax reports", "multi-currency support"],
    "HR Software": ["onboarding checklists", "PTO tracking", "performance reviews", "org chart", "payroll export"],
    "Analytics": ["dashboards", "custom reports", "data connectors", "scheduled exports", "funnel analysis"],
}

ROLES = ["Sales Manager", "Project Manager", "Marketing Specialist", "Operations Lead", "CTO",
         "Customer Success Manager", "Founder", "Finance Manager", "HR Generalist", "Data Analyst",
         "Office Manager", "Product Owner", "Support Team Lead", "Account Executive"]
SIZES = ["1-10 employees", "11-50 employees", "51-200 employees", "201-500 employees", "501-1000 employees", "1000+ employees"]

POS_OPEN = [
    "We switched to {p} last year and haven't looked back.",
    "{p} has become a core part of our daily workflow.",
    "I've used {p} for about two years now.",
    "Our team adopted {p} after trying three other tools.",
    "Really happy with {p} overall.",
    "{p} does exactly what we needed it to do.",
    "We rolled out {p} to the whole team in under a week.",
    "Been a {p} customer for 18 months.",
    "",
]
NEG_OPEN = [
    "We moved to {p} last year and regret it.",
    "I wanted to like {p}, but it has been a struggle.",
    "We used {p} for about eight months before cancelling.",
    "Our team tried {p} and it did not work out.",
    "Pretty disappointed with {p} overall.",
    "{p} looked great in the demo, the reality is different.",
    "We are currently looking for an alternative to {p}.",
    "Used {p} for a year, not renewing.",
    "",
]

POS_SENT = [
    "The {f} is intuitive and saves us hours every week.",
    "Setup was quick and the onboarding guides were clear.",
    "Customer support answers within minutes and actually solves the problem.",
    "The interface is clean and easy to learn, even for non-technical people.",
    "Pricing is fair for what you get.",
    "It integrates smoothly with Slack and Google Workspace.",
    "The {f} works exactly as advertised.",
    "Reports are easy to customize and share with management.",
    "The mobile app is surprisingly good.",
    "It's fast, even with thousands of records.",
    "New features ship regularly and they listen to feedback.",
    "Our team adoption was almost instant.",
    "The {f} alone is worth the price.",
    "Documentation is thorough and up to date.",
    "Reliable, we've had basically no downtime.",
    "The API is well designed and easy to work with.",
    "Our account manager is responsive and genuinely helpful.",
    "Automations cut a lot of repetitive manual work.",
    "It replaced two other tools for us, which saved money.",
    "Permissions and roles are flexible enough for our structure.",
    "Not cheap, but definitely worth it.",
    "I can't imagine going back to spreadsheets.",
    "Not a single bug in the {f} so far.",
    "No complaints about the {f}.",
    "Never had an issue with billing.",
    "Not bad at all once you get used to it.",
    "Our sales calls are shorter because everything is in one place.",
    "The free trial convinced the whole team.",
    "Search actually finds what I'm looking for.",
    "Importing our old data took one afternoon.",
    "The templates gave us a head start.",
    "Hard to find anything they could improve.",
    "Our managers finally have visibility into what everyone is doing.",
]
NEG_SENT = [
    "The {f} is clunky and slow to load.",
    "Setup took weeks and the onboarding was confusing.",
    "Customer support takes days to reply and rarely fixes anything.",
    "The interface is cluttered and hard to navigate.",
    "Way too expensive for what you get.",
    "The Slack integration breaks every few weeks.",
    "The {f} doesn't work as advertised.",
    "Reporting is very limited and hard to customize.",
    "The mobile app crashes constantly.",
    "It gets painfully slow once you have a few thousand records.",
    "They raised the price by 40% with almost no notice.",
    "Half of our team refused to use it.",
    "The {f} is buggy and loses data occasionally.",
    "Documentation is outdated and missing key details.",
    "We had several outages during business hours.",
    "The API is poorly documented and rate limits are strict.",
    "Our account manager disappeared after we signed the contract.",
    "Automations fail silently, so you never know what broke.",
    "Basic features are locked behind the enterprise plan.",
    "Permissions are too rigid for a team our size.",
    "Not worth the money at all.",
    "Honestly, our old spreadsheet worked better.",
    "Cancelling was a nightmare, they kept charging us.",
    "The {f} is not intuitive at all.",
    "It never syncs correctly with our calendar.",
    "Nothing about the {f} feels finished.",
    "Our sales calls got longer because the data is scattered.",
    "The free trial hid most of the problems.",
    "Search almost never finds what I'm looking for.",
    "Importing our old data took three weeks and a consultant.",
    "The templates are generic and mostly useless.",
    "Hard to find anything they do well.",
    "Our managers still export everything to Excel.",
]

# Small "con" in a positive review / "pro" in a negative review, to make it realistic
MINOR_CON = [
    "The only downside is that the {f} could be a bit faster.",
    "My one complaint is the price for larger teams.",
    "The mobile app could use some work, but it's fine.",
    "Reporting could be more flexible.",
    "There is a small learning curve at the start.",
]
MINOR_PRO = [
    "To be fair, the {f} looks nice.",
    "The interface is pretty, I'll give them that.",
    "Support staff are friendly, just not effective.",
    "The idea behind the product is good.",
    "Setup itself was easy enough.",
]

NEUTRAL = [
    "We are a team of {n} people.",
    "We mainly use it for the {f}.",
    "I use it every day.",
    "We evaluated it against two competitors.",
    "Our IT team handled the rollout.",
    "We are on the business plan.",
    "Most of our team works remotely.",
    "We migrated from a legacy system.",
]
NEUTRAL_TITLE = ["Our experience with {p}", "{p} review", "Two years in", "Honest review",
                 "Thoughts after one year", "{p} for a mid-sized team", "My take on {p}"]

POS_CLOSE = ["Highly recommend.", "Would recommend to any growing team.", "5 stars from me.",
             "Great value.", "Definitely renewing.", "Solid choice.", ""]
NEG_CLOSE = ["Would not recommend.", "Look elsewhere.", "Avoid if you can.",
             "Not renewing.", "Very frustrating experience.", "Save your money.", ""]

POS_TITLE = ["Great tool for our team", "Does the job really well", "Huge time saver", "Love it",
             "Best {c} tool we've tried", "Easy to use and reliable", "Worth every penny", "Solid and fast"]
NEG_TITLE = ["Frustrating experience", "Not worth the price", "Too many bugs", "Disappointed",
             "Avoid this {c} tool", "Support is non-existent", "Looked good, isn't", "Slow and clunky"]


def fill(s, p, f, c):
    return s.format(p=p, f=f, c=c)


def make_review(label, cat, prod):
    feats = FEATURE[cat]
    if label == "positive":
        opener, body, other, close, title = POS_OPEN, POS_SENT, NEG_SENT, POS_CLOSE, POS_TITLE
        rating = random.choices([4, 5], weights=[5, 5])[0]
        minor = MINOR_CON
    else:
        opener, body, other, close, title = NEG_OPEN, NEG_SENT, POS_SENT, NEG_CLOSE, NEG_TITLE
        rating = random.choices([1, 2], weights=[5, 5])[0]
        minor = MINOR_PRO

    def f():
        return random.choice(feats)

    parts = []
    if random.random() < 0.5:
        parts.append(random.choice(opener).format(p=prod))
    if random.random() < 0.4:
        parts.append(random.choice(NEUTRAL).format(n=random.randint(4, 300), f=f()))
    main = [random.choice(body).format(f=f()) for _ in range(random.randint(1, 3))]
    # mixed opinions: the overall verdict decides the label, not every sentence
    r = random.random()
    if r < 0.30:
        main.insert(random.randint(0, len(main)), random.choice(minor).format(f=f()))
    elif r < 0.55:
        main.insert(random.randint(0, len(main)), random.choice(other).format(f=f()))
    parts += list(dict.fromkeys(main))
    if random.random() < 0.6:
        parts.append(random.choice(close))
    text = " ".join(p for p in parts if p).strip()
    if random.random() < 0.45:
        t = random.choice(NEUTRAL_TITLE).format(p=prod)
    else:
        t = random.choice(title).format(c=cat.lower())
    return t, text, rating


rows = []
start = date(2024, 1, 1)
N = 2000
for i in range(1, N + 1):
    cat = random.choice(list(PRODUCTS))
    prod = random.choice(PRODUCTS[cat])
    label = random.choices(["positive", "negative"], weights=[6, 4])[0]  # real sites skew positive
    title, text, rating = make_review(label, cat, prod)
    if random.random() < 0.03:  # realistic noise: some ratings don't match the text
        label = "negative" if label == "positive" else "positive"
        rating = 2 if label == "negative" else 4
    rows.append({
        "review_id": i,
        "product": prod,
        "category": cat,
        "reviewer_role": random.choice(ROLES),
        "company_size": random.choice(SIZES),
        "date": (start + timedelta(days=random.randint(0, 640))).isoformat(),
        "rating": rating,
        "title": title,
        "review_text": text,
        "label": label,
    })

# Leave the label empty for the last 100 rows: practice predicting on "new" reviews
for r in rows[-100:]:
    r["label"] = ""

with open("./data/tech_product_reviews_labeled.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("wrote", len(rows), "rows")