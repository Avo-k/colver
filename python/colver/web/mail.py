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

**Un SMTP configuré et joignable ne veut pas dire qu'un courriel arrive.** Le
relais peut accepter la transaction (`250`) puis jeter le message — Mailjet le
fait quand l'expéditeur du `From` n'est pas validé sur le compte, et ne prévient
que par un courriel à l'administrateur. Ce module ne peut pas voir ça : il
journalise donc la réponse du relais, identifiant de file compris, seule prise
pour retrouver le message chez lui ensuite. `/health` publie `mail.configured`,
qui est tout ce qu'on sache sans envoyer.

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


class _Traced:
    """Retenir la réponse du relais au `DATA` — `sendmail` la jette.

    C'est la **seule** chose que le serveur nous dise du sort du message, et
    elle porte en général un identifiant de file (Mailjet : « OK queued as
    <uuid> ») avec lequel on peut ensuite retrouver ce qu'il en a fait. Sans
    elle, un message accepté puis jeté par le relais est indiscernable d'un
    message remis — c'est exactement ce qui a masqué trois jours de panne le
    2026-08-05, l'expéditeur `no-reply@colver.net` n'étant pas encore validé
    côté Mailjet : `250 OK queued`, journal rassurant, aucune remise.
    """

    data_reply = None

    def data(self, msg):
        code, resp = super().data(msg)
        if isinstance(resp, bytes):
            resp_text = resp.decode("utf-8", "replace")
        else:
            resp_text = str(resp)
        self.data_reply = resp_text
        return code, resp


class _SMTP(_Traced, smtplib.SMTP):
    pass


class _SMTP_SSL(_Traced, smtplib.SMTP_SSL):
    pass


def enabled() -> bool:
    return bool(SMTP_HOST)


def status():
    """Ce que `/health` publie du courriel. **Ni identifiant ni mot de passe.**

    `configured` est la seule chose qu'on sache sans envoyer : le reste — le
    relais accepte-t-il notre expéditeur, remet-il vraiment — ne se découvre
    qu'à l'envoi, et se lit dans le journal (cf. `send`) ou chez le relais.
    """
    return {
        "configured": enabled(),
        "host": SMTP_HOST or None,
        "sender": _sender() if enabled() else None,
    }


def _sender() -> str:
    if MAIL_FROM:
        return MAIL_FROM
    host = SMTP_HOST or "colver.net"
    return f"colver <no-reply@{host}>"


def send(to, subject, body):
    """Confier un message au relais. Synchrone — à appeler via `asyncio.to_thread`.

    Rend True si le relais l'a **accepté**, False s'il a refusé ou si aucun SMTP
    n'est configuré. **Ne lève jamais** : l'appelant répond la même chose dans
    tous les cas, et une panne d'envoi ne doit pas devenir une 500 qui, elle,
    dirait au visiteur que son adresse existe bien.

    **Accepté n'est pas remis, et ce True ne prétend pas le contraire.** Le
    protocole s'arrête au relais : la suite (expéditeur autorisé, DMARC, boîte
    du destinataire) se joue après, sans nous. D'où la réponse du serveur dans
    le journal — elle porte de quoi retrouver le message chez le relais quand
    quelqu'un dit ne rien avoir reçu.
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
            client = _SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT,
                               context=ssl.create_default_context())
        else:
            client = _SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT)
        with client:
            if SMTP_TLS == "starttls":
                client.starttls(context=ssl.create_default_context())
            if SMTP_USER:
                client.login(SMTP_USER, SMTP_PASSWORD)
            client.send_message(msg)
            reply = client.data_reply
        logger.info("courriel accepté par le relais pour %s (%s) — %s",
                    to, subject, reply or "sans réponse")
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
