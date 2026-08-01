"""Envoi de courriel — un seul usage aujourd'hui : le lien de réinitialisation.

**Sans configuration SMTP, le lien part au journal au lieu de la boîte mail.**
C'est délibéré et ça vaut pour le développement comme pour un déploiement où
l'envoi n'est pas encore branché : la fonctionnalité marche de bout en bout
(jeton créé, lien valable, consommation unique), il faut juste aller chercher le
lien dans les logs. Le contraire — refuser la demande faute de SMTP — rendrait
la réinitialisation intestable et masquerait les vraies pannes d'envoi.

L'appelant ne voit pas la différence : `send` rend toujours quelque chose, et la
réponse HTTP est de toute façon la même dans tous les cas (cf. `auth.forgot` —
elle ne doit pas révéler si une adresse existe).

Configuration (toutes facultatives) :
  COLVER_SMTP_HOST / _PORT (587) / _USER / _PASSWORD
  COLVER_SMTP_TLS      "starttls" (défaut) | "ssl" | "none"
  COLVER_MAIL_FROM     expéditeur ; défaut « colver <no-reply@[host]> »
  COLVER_PUBLIC_URL    déjà utilisé par le SEO ; sert à fabriquer les liens
"""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)

SMTP_HOST = os.environ.get("COLVER_SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("COLVER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("COLVER_SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("COLVER_SMTP_PASSWORD", "")
SMTP_TLS = os.environ.get("COLVER_SMTP_TLS", "starttls").strip().lower()
MAIL_FROM = os.environ.get("COLVER_MAIL_FROM", "").strip()

# Délai court et assumé : l'envoi tourne dans un thread, mais la requête HTTP
# l'attend. Un serveur SMTP qui ne répond pas ne doit pas retenir le joueur.
SMTP_TIMEOUT = 10


def enabled() -> bool:
    return bool(SMTP_HOST)


def _sender() -> str:
    if MAIL_FROM:
        return MAIL_FROM
    host = SMTP_HOST or "colver.net"
    return f"colver <no-reply@{host}>"


def send(to, subject, body):
    """Envoyer un message. Synchrone — à appeler via `asyncio.to_thread`.

    Rend True si le message est parti, False s'il a échoué ou si aucun SMTP
    n'est configuré. **Ne lève jamais** : l'appelant répond la même chose dans
    tous les cas, et une panne d'envoi ne doit pas devenir une 500 qui, elle,
    dirait au visiteur que son adresse existe bien.
    """
    if not enabled():
        logger.warning(
            "SMTP non configuré — courriel non envoyé à %s.\n"
            "--- %s ---\n%s\n---", to, subject, body)
        return False

    msg = EmailMessage()
    msg["From"] = _sender()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if SMTP_TLS == "ssl":
            client = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT,
                                      context=ssl.create_default_context())
        else:
            client = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT)
        with client:
            if SMTP_TLS == "starttls":
                client.starttls(context=ssl.create_default_context())
            if SMTP_USER:
                client.login(SMTP_USER, SMTP_PASSWORD)
            client.send_message(msg)
        logger.info("courriel envoyé à %s (%s)", to, subject)
        return True
    except Exception:
        logger.exception("envoi du courriel à %s en échec", to)
        return False


def reset_email(username, link, hours):
    """Le corps du message de réinitialisation. Séparé pour être testable.

    Volontairement sec, et **sans rien affirmer sur qui a demandé** : le
    formulaire est public, donc une adresse peut recevoir ce message sans que
    son propriétaire ait rien fait. La dernière phrase est là pour ça.
    """
    subject = "colver — réinitialiser votre mot de passe"
    body = (
        f"Bonjour {username},\n\n"
        f"Pour choisir un nouveau mot de passe, suivez ce lien :\n\n"
        f"    {link}\n\n"
        f"Il est valable {hours} h et ne fonctionne qu'une fois.\n\n"
        f"Si vous n'avez rien demandé, ignorez ce message : votre mot de passe\n"
        f"actuel reste valable et n'a pas été modifié.\n"
    )
    return subject, body
