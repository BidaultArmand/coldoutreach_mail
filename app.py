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

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

load_dotenv()

BACKLOG_FILE = Path("backlog.csv")
BACKLOG_COLS = ["nom", "email", "statut"]
STATUS_PENDING = "En attente"
STATUS_SENT = "Envoyé"
STATUS_ERROR = "Erreur"


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
            "statut": STATUS_PENDING,
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
        '[{"nom": "Nom de l\'établissement", "email": "contact@example.com"}, ...]. '
        "Ne propose que des emails réels trouvés sur le web (contact@, info@, reservation@…). "
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
# Session state init
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "sender_email": os.getenv("SENDER_EMAIL", ""),
        "app_password": os.getenv("APP_PASSWORD", ""),
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "mail_subject": "Partenariat – Une opportunité pour {nom}",
        "mail_body": (
            "Bonjour,\n\n"
            "Je me permets de vous contacter au sujet de {nom}.\n\n"
            "...\n\n"
            "Cordialement,"
        ),
        "chat_history": [],
        "target_count": 10,
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

        st.session_state.chat_history.append({
            "role": "user",
            "display": user_input,
            "raw_content": user_input,
        })

        with st.spinner("Gemini recherche via Google Search…"):
            try:
                df_ctx = load_backlog()
                existing = df_ctx[["nom", "email"]].to_dict("records")

                raw = search_prospects_with_gemini_chat(
                    st.session_state.gemini_key,
                    st.session_state.chat_history[:-1],  # history without latest user msg
                    user_input,
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("En attente", pending)
    col3.metric("Envoyés", sent)

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

    # Per-row table with action buttons
    # Use column headers manually
    header_cols = st.columns([3, 3, 2, 2])
    header_cols[0].markdown("**Nom**")
    header_cols[1].markdown("**Email**")
    header_cols[2].markdown("**Statut**")
    header_cols[3].markdown("**Action**")

    for _, row in df.iterrows():
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        c1.write(row["nom"])
        c2.write(row["email"])

        statut = row["statut"]
        if statut == STATUS_SENT:
            c3.success(statut)
        elif statut == STATUS_PENDING:
            c3.warning(statut)
        else:
            c3.error(statut)

        if statut != STATUS_SENT:
            if c4.button("Envoyer", key=f"send_{row['email']}"):
                df = _send_one(row, df)
                st.rerun()
        else:
            c4.write("—")


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
