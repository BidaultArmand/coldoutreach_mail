import streamlit as st
import pandas as pd
import smtplib
import json
import re
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
import dns.resolver
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

load_dotenv()

BACKLOG_FILE = Path("backlog.csv")
BACKLOG_COLS = ["nom", "email", "site", "statut", "verification"]
STATUS_PENDING = "En attente"
STATUS_SENT = "Envoyé"
STATUS_ERROR = "Erreur"

VERIF_NONE = ""
VERIF_DNS_OK = "DNS OK"
VERIF_DNS_FAIL = "Domaine invalide"
VERIF_VALID = "Valide"
VERIF_DOUBTFUL = "Douteux"
VERIF_INVALID = "Invalide"


# ---------------------------------------------------------------------------
# Backlog helpers
# ---------------------------------------------------------------------------

def load_backlog() -> pd.DataFrame:
    if BACKLOG_FILE.exists():
        df = pd.read_csv(BACKLOG_FILE, dtype=str)
        for col in BACKLOG_COLS:
            if col not in df.columns:
                df[col] = ""
        return df[BACKLOG_COLS].fillna("")
    return pd.DataFrame(columns=BACKLOG_COLS)


def save_backlog(df: pd.DataFrame) -> None:
    df.to_csv(BACKLOG_FILE, index=False)


def add_prospects(new_rows: list[dict]) -> tuple[int, int]:
    """Add prospects avoiding duplicates by email. Returns (added, skipped)."""
    df = load_backlog()
    existing_emails = set(df["email"].str.lower())
    added, skipped = 0, 0
    for row in new_rows:
        email = row.get("email", "").strip().lower()
        if not email or email in existing_emails:
            skipped += 1
            continue
        entry = {
            "nom": row.get("nom", "").strip(),
            "email": email,
            "site": row.get("site", "").strip(),
            "statut": STATUS_PENDING,
            "verification": VERIF_NONE,
        }
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        existing_emails.add(email)
        added += 1
    save_backlog(df)
    return added, skipped


# ---------------------------------------------------------------------------
# Gemini search
# ---------------------------------------------------------------------------

def search_prospects_with_gemini_chat(
    api_key: str,
    history: list[dict],
    user_message: str,
    target_count: int,
    existing_prospects: list[dict],
) -> str:
    """
    Multi-turn chat with Gemini + Google Search Grounding.
    Returns the raw response text (JSON or plain text).
    """
    client = genai.Client(api_key=api_key)

    backlog_lines = (
        "\n".join(f"- {p['nom']} ({p['email']})" for p in existing_prospects)
        if existing_prospects
        else "Aucun pour l'instant."
    )

    system_instruction = (
        "Tu es un assistant de prospection commerciale. "
        f"L'objectif est d'atteindre {target_count} prospects au total dans le backlog. "
        "Quand l'utilisateur te demande de trouver ou chercher des établissements, utilise Google Search "
        "et réponds UNIQUEMENT avec un tableau JSON valide (sans markdown, sans texte autour) : "
        '[{"nom": "Nom de l\'établissement", "email": "contact@example.com", "site": "https://..."}, ...]. '
        "Ne propose que des emails réels trouvés sur le web (contact@, info@, reservation@…). "
        "Inclus toujours l'URL du site officiel de l'établissement dans le champ 'site'. "
        "Si tu ne trouves pas l'email d'un établissement, ne l'inclus pas. "
        "Pour toute autre demande (question, conseil, précision), réponds normalement en français. "
        f"\n\nProspects DÉJÀ dans le backlog (ne pas re-proposer) :\n{backlog_lines}"
    )

    # Rebuild typed history for the chat SDK
    chat_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        chat_history.append(
            types.Content(role=role, parts=[types.Part(text=msg["raw_content"])])
        )

    chat = client.chats.create(
        model="gemini-2.5-flash",
        history=chat_history,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    response = chat.send_message(user_message)
    return response.text


# ---------------------------------------------------------------------------
# Email sending
# ---------------------------------------------------------------------------

def send_email(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    app_password: str,
    recipient_email: str,
    subject: str,
    body: str,
) -> None:
    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


def format_body(template: str, nom: str) -> str:
    return template.replace("{nom}", nom).replace("{nom_hotel}", nom)


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------

def verify_email_dns(email: str) -> tuple[bool, str]:
    """Check email format then MX/A records for its domain."""
    if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email):
        return False, "Format d'email invalide"
    domain = email.split("@")[1]
    try:
        dns.resolver.resolve(domain, "MX")
        return True, f"Domaine {domain} OK (MX)"
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
        try:
            dns.resolver.resolve(domain, "A")
            return True, f"Domaine {domain} OK (A)"
        except Exception:
            return False, f"Domaine {domain} introuvable"
    except Exception as e:
        return False, f"Erreur DNS : {e}"


def verify_email_with_openai(openai_key: str, nom: str, email: str) -> tuple[str, str]:
    """
    Use GPT-4o with web search to check that the email really belongs
    to the given establishment. Returns (status, explanation).
    """
    client = OpenAI(api_key=openai_key)

    prompt = (
        f"Vérifie si l'adresse email '{email}' est bien l'email de contact officiel "
        f"de l'établissement '{nom}'. Cherche leur site web officiel. "
        "Réponds UNIQUEMENT en JSON sans markdown : "
        '{"statut": "valide" | "douteux" | "invalide", "explication": "une phrase courte"}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-search-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    result = json.loads(raw)
    statut = result.get("statut", "douteux")
    explication = result.get("explication", "")
    return statut, explication


def verify_one(row: pd.Series, df: pd.DataFrame, openai_key: str) -> pd.DataFrame:
    """Run DNS check, then OpenAI check if key is provided. Updates df in place."""
    email = row["email"]
    nom = row["nom"]

    dns_ok, dns_msg = verify_email_dns(email)
    if not dns_ok:
        verif = VERIF_DNS_FAIL
    elif not openai_key:
        verif = VERIF_DNS_OK
    else:
        try:
            statut, _ = verify_email_with_openai(openai_key, nom, email)
            verif = {"valide": VERIF_VALID, "invalide": VERIF_INVALID}.get(statut, VERIF_DOUBTFUL)
        except Exception:
            verif = VERIF_DNS_OK  # fall back to DNS result on OpenAI error

    df.loc[df["email"] == email, "verification"] = verif
    save_backlog(df)
    return df


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "sender_email": os.getenv("SENDER_EMAIL", ""),
        "app_password": os.getenv("APP_PASSWORD", ""),
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "mail_subject": "Recherche académique – échange rapide sur les opérations ",
        "mail_body": (
            "Bonjour,\n\n"
            "Je suis actuellement étudiant au sein du Master X-HEC Entrepreneurs à HEC Paris et je mène une étude académique "
            "portant sur l’expérience client et les coulisses opérationnelles des hôtels à forte identité.\n\n"
            "{nom_hotel} correspond précisément au type de modèle que j’analyse dans le cadre de mes recherches.\n"
            "J’aimerais beaucoup pouvoir échanger 15 minutes avec vous, le Directeur Général ou toute personne disposant "
            "d’une vision globale des opérations de l’établissement.\n\n"
            "Il s’agit exclusivement d’un travail académique, sans aucune démarche commerciale. "
            "Si un autre interlocuteur vous semble plus pertinent sur ce sujet, je vous serais reconnaissant de bien vouloir "
            "me rediriger vers la bonne personne.\n\n"
            "Un court échange serait déjà très précieux pour la qualité et la précision de mon étude.\n"
            "Je vous remercie par avance pour votre temps et votre considération.\n\n"
            "Bien cordialement,\n"
            "Armand Bidault"
        ),
        "chat_history": [],
        "target_count": 10,
        "selected_tags": [],
        "openai_key": os.getenv("OPENAI_API_KEY", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def sidebar_config():
    st.sidebar.header("Configuration")

    st.session_state.gemini_key = st.sidebar.text_input(
        "Clé API Gemini",
        value=st.session_state.gemini_key,
        type="password",
        help="Disponible sur https://aistudio.google.com/app/apikey",
    )
    st.session_state.sender_email = st.sidebar.text_input(
        "Email expéditeur",
        value=st.session_state.sender_email,
    )
    st.session_state.app_password = st.sidebar.text_input(
        "Mot de passe d'application",
        value=st.session_state.app_password,
        type="password",
        help="Pour Gmail : Compte Google > Sécurité > Mots de passe des applications",
    )
    st.session_state.smtp_server = st.sidebar.text_input(
        "Serveur SMTP",
        value=st.session_state.smtp_server,
    )
    st.session_state.smtp_port = st.sidebar.number_input(
        "Port SMTP",
        value=st.session_state.smtp_port,
        min_value=1,
        max_value=65535,
        step=1,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Vérification des emails")
    st.session_state.openai_key = st.sidebar.text_input(
        "Clé API OpenAI (optionnel)",
        value=st.session_state.openai_key,
        type="password",
        help="Si renseignée, active la vérification via GPT-4o + web search. Sans clé, seul le check DNS est effectué.",
    )

    st.sidebar.divider()
    if st.sidebar.button("Vider le backlog", type="secondary", use_container_width=True):
        save_backlog(pd.DataFrame(columns=BACKLOG_COLS))
        st.sidebar.success("Backlog vidé.")
        st.rerun()


def section_search():
    st.subheader("Recherche de prospects")

    # --- Controls row ---
    col_prog, col_num = st.columns([4, 1])
    df_now = load_backlog()
    current = len(df_now)
    target = st.session_state.target_count

    with col_num:
        st.session_state.target_count = st.number_input(
            "Prospects cibles",
            min_value=1,
            max_value=500,
            value=target,
            step=1,
            help="Nombre total de prospects à atteindre dans le backlog.",
        )
        target = st.session_state.target_count

    with col_prog:
        pct = min(current / target, 1.0) if target else 0
        st.progress(pct, text=f"{current} / {target} prospects dans le backlog")

    st.divider()

    # --- Context tag buttons ---
    CONTEXT_TAGS = [
        "Lifestyle",
        "Design & architecture",
        "Expérientiel",
        "Premium",
        "Haut de gamme",
        "Nouveauté",
    ]
    st.caption("Contexte rapide (cliquer pour sélectionner / désélectionner) :")
    tag_cols = st.columns(len(CONTEXT_TAGS))
    for i, tag in enumerate(CONTEXT_TAGS):
        selected = tag in st.session_state.selected_tags
        label = f"✓ {tag}" if selected else tag
        btn_type = "primary" if selected else "secondary"
        if tag_cols[i].button(label, key=f"tag_{tag}", type=btn_type, use_container_width=True):
            if selected:
                st.session_state.selected_tags.remove(tag)
            else:
                st.session_state.selected_tags.append(tag)
            st.rerun()

    st.divider()

    # --- Chat history display ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["display"])

    # --- Chat input ---
    user_input = st.chat_input(
        "Ex : Trouve 10 hôtels lifestyle 4* dans le 13e à Paris…"
    )

    if user_input:
        if not st.session_state.gemini_key:
            st.error("Renseigne ta clé API Gemini dans la sidebar.")
            return

        # Append selected tags to the prompt sent to Gemini
        tags = st.session_state.selected_tags
        if tags:
            tags_str = ", ".join(tags)
            enriched_input = f"{user_input}\n\nContexte additionnel : {tags_str}"
            display_input = f"{user_input}\n\n*Filtres actifs : {tags_str}*"
        else:
            enriched_input = user_input
            display_input = user_input

        # Clear tags after sending
        st.session_state.selected_tags = []

        st.session_state.chat_history.append({
            "role": "user",
            "display": display_input,
            "raw_content": enriched_input,
        })

        with st.spinner("Gemini recherche via Google Search…"):
            try:
                df_ctx = load_backlog()
                existing = df_ctx[["nom", "email"]].to_dict("records")

                raw = search_prospects_with_gemini_chat(
                    st.session_state.gemini_key,
                    st.session_state.chat_history[:-1],  # history without latest user msg
                    enriched_input,
                    st.session_state.target_count,
                    existing,
                )

                # Try to parse as JSON list of prospects
                clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                clean = re.sub(r"\s*```$", "", clean.strip())

                try:
                    prospects = json.loads(clean)
                    if not isinstance(prospects, list):
                        raise ValueError

                    added, skipped = add_prospects(prospects)
                    rows_md = "\n".join(
                        f"- **{p.get('nom', '?')}** — `{p.get('email', '?')}`"
                        + (f" — [{p['site']}]({p['site']})" if p.get('site') else "")
                        for p in prospects
                    )
                    display = (
                        f"**{len(prospects)} prospect(s) trouvé(s)** "
                        f"— {added} ajouté(s) au backlog, {skipped} ignoré(s) (doublons / email manquant).\n\n"
                        + rows_md
                    )

                except (json.JSONDecodeError, ValueError):
                    display = raw

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "display": display,
                    "raw_content": raw,
                })

            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "display": f"Erreur : {e}",
                    "raw_content": str(e),
                })

        st.rerun()

    if st.session_state.chat_history:
        if st.button("Effacer la conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


def section_template():
    st.subheader("Template d'email")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.mail_subject = st.text_input(
            "Objet",
            value=st.session_state.mail_subject,
        )
    with col2:
        st.caption("Variables disponibles : `{nom}` ou `{nom_hotel}`")

    st.session_state.mail_body = st.text_area(
        "Corps du message",
        value=st.session_state.mail_body,
        height=200,
    )


def _send_one(row: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Send a single email and update the backlog in place."""
    cfg = st.session_state
    required = [cfg.sender_email, cfg.app_password, cfg.smtp_server]
    if not all(required):
        st.error("Configure les paramètres SMTP dans la sidebar.")
        return df

    nom = row["nom"]
    email = row["email"]
    subject = format_body(cfg.mail_subject, nom)
    body = format_body(cfg.mail_body, nom)

    try:
        send_email(
            cfg.smtp_server,
            int(cfg.smtp_port),
            cfg.sender_email,
            cfg.app_password,
            email,
            subject,
            body,
        )
        df.loc[df["email"] == email, "statut"] = STATUS_SENT
    except Exception as e:
        df.loc[df["email"] == email, "statut"] = f"{STATUS_ERROR}: {e}"

    save_backlog(df)
    return df


def section_backlog():
    st.subheader("Backlog des prospects")

    df = load_backlog()

    if df.empty:
        st.info("Le backlog est vide. Lance une recherche pour ajouter des prospects.")
        return

    # Summary metrics
    total = len(df)
    sent = (df["statut"] == STATUS_SENT).sum()
    pending = (df["statut"] == STATUS_PENDING).sum()
    verified = df["verification"].isin([VERIF_VALID, VERIF_DNS_OK]).sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("En attente", pending)
    col3.metric("Envoyés", sent)
    col4.metric("Vérifiés", verified)

    # "Tout envoyer" button
    pending_df = df[df["statut"] == STATUS_PENDING]
    if not pending_df.empty:
        if st.button(
            f"Tout envoyer ({len(pending_df)} mails)",
            type="primary",
            use_container_width=True,
        ):
            progress = st.progress(0, text="Envoi en cours...")
            for i, (_, row) in enumerate(pending_df.iterrows()):
                df = _send_one(row, df)
                progress.progress(
                    (i + 1) / len(pending_df),
                    text=f"Envoyé {i + 1}/{len(pending_df)} — {row['nom']}",
                )
                if i < len(pending_df) - 1:
                    time.sleep(2)
            progress.empty()
            st.success("Envoi terminé.")
            st.rerun()

    st.divider()

    # "Tout vérifier" button
    unverified = df[df["verification"] == VERIF_NONE]
    verif_label = (
        "Tout vérifier (DNS + OpenAI)" if st.session_state.openai_key
        else "Tout vérifier (DNS uniquement)"
    )
    if not unverified.empty:
        if st.button(f"{verif_label} — {len(unverified)} prospect(s)", use_container_width=True):
            progress = st.progress(0, text="Vérification en cours…")
            for i, (_, row) in enumerate(unverified.iterrows()):
                df = verify_one(row, df, st.session_state.openai_key)
                progress.progress(
                    (i + 1) / len(unverified),
                    text=f"Vérifié {i + 1}/{len(unverified)} — {row['nom']}",
                )
            progress.empty()
            st.success("Vérification terminée.")
            st.rerun()

    st.divider()

    # Per-row table with action buttons
    header_cols = st.columns([3, 3, 2, 2, 2, 2])
    header_cols[0].markdown("**Nom**")
    header_cols[1].markdown("**Email**")
    header_cols[2].markdown("**Site**")
    header_cols[3].markdown("**Vérification**")
    header_cols[4].markdown("**Statut**")
    header_cols[5].markdown("**Actions**")

    for _, row in df.iterrows():
        c1, c2, c3, c4, c5, c6 = st.columns([3, 3, 2, 2, 2, 2])
        c1.write(row["nom"])
        c2.write(row["email"])

        site = row.get("site", "")
        if site:
            c3.markdown(f"[Ouvrir]({site})")
        else:
            c3.write("—")

        verif = row.get("verification", VERIF_NONE)
        if verif == VERIF_VALID:
            c4.success("✓ Valide")
        elif verif == VERIF_DNS_OK:
            c4.info("DNS OK")
        elif verif in (VERIF_INVALID, VERIF_DNS_FAIL):
            c4.error(verif)
        elif verif == VERIF_DOUBTFUL:
            c4.warning("⚠ Douteux")
        else:
            if c4.button("Vérifier", key=f"verif_{row['email']}"):
                df = verify_one(row, df, st.session_state.openai_key)
                st.rerun()

        statut = row["statut"]
        if statut == STATUS_SENT:
            c5.success(statut)
        elif statut == STATUS_PENDING:
            c5.warning(statut)
        else:
            c5.error(statut)

        if statut != STATUS_SENT:
            if c6.button("Envoyer", key=f"send_{row['email']}"):
                df = _send_one(row, df)
                st.rerun()
        else:
            c6.write("—")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Customer Discovery Outreach",
        page_icon="envelope",
        layout="wide",
    )
    init_session()
    sidebar_config()

    st.title("Customer Discovery — Automatisation des emails")

    tab_search, tab_template, tab_backlog = st.tabs(
        ["Recherche", "Template email", "Backlog / Envoi"]
    )

    with tab_search:
        section_search()

    with tab_template:
        section_template()

    with tab_backlog:
        section_backlog()


if __name__ == "__main__":
    main()
