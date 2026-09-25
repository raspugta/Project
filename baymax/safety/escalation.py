"""Configurable emergency escalation.

Only the safety machine's `notify_contacts` action reaches `dispatch()`.
Alert text is fixed and minimal: it never contains conversation text, and it
contains the triggering health reason only with the
`share_health_details_in_alerts` consent. Dry-run is ON by default so a new
installation never pages anyone until the user has tested and switched it off.
"""
from __future__ import annotations

import os
import smtplib
import time
from email.message import EmailMessage
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

from ..contracts import new_id
from ..ontology import SUPPORTED_LANGS

Channel = Literal["webhook", "email", "console"]


class Contact(BaseModel):
    id: str = Field(default_factory=lambda: new_id("contact"))
    name: str
    channel: Channel = "webhook"
    address: str                 # webhook URL / e-mail address / label for console
    lang: str = "en"


class EscalationConfig(BaseModel):
    emergency_number: str = "112"
    dry_run: bool = True
    contacts: list[Contact] = []
    user_display_name: str = "Baymax user"
    webhook_timeout_s: float = 5.0


class ChannelResult(BaseModel):
    contact_id: str
    channel: Channel
    ok: bool
    detail: str = ""


class EscalationResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("esc"))
    status: Literal["sent", "partial", "failed", "dry_run", "no_contacts", "no_consent"]
    level: str
    reason: str
    results: list[ChannelResult] = []
    ts: float = Field(default_factory=time.time)


ALERT: dict[str, dict[str, str]] = {
    "emergency": {
        "en": "BAYMAX ALERT: {user} may need help right now. Please check on them immediately.",
        "hi": "बेमैक्स अलर्ट: {user} को अभी मदद की ज़रूरत हो सकती है। कृपया तुरंत उनका हालचाल लें।",
        "hi-Latn": "BAYMAX ALERT: {user} ko abhi madad ki zaroorat ho sakti hai. Please turant unka haal-chaal lijiye.",
        "es": "ALERTA BAYMAX: {user} puede necesitar ayuda ahora mismo. Por favor, compruébalo de inmediato.",
        "fr": "ALERTE BAYMAX : {user} a peut-être besoin d'aide maintenant. Merci de prendre de ses nouvelles immédiatement.",
        "de": "BAYMAX-ALARM: {user} braucht möglicherweise sofort Hilfe. Bitte sieh sofort nach.",
    },
    "urgent": {
        "en": "BAYMAX: {user} asked me to let you know they need someone to check on them.",
        "hi": "बेमैक्स: {user} ने मुझसे कहा कि आपको बताऊं कि उन्हें किसी के हालचाल लेने की ज़रूरत है।",
        "hi-Latn": "BAYMAX: {user} ne mujhse kaha ki aapko bataoon ki unhe kisi ke haal-chaal lene ki zaroorat hai.",
        "es": "BAYMAX: {user} me pidió que te avisara de que necesita que alguien vaya a verle.",
        "fr": "BAYMAX : {user} m'a demandé de vous prévenir qu'il/elle a besoin que quelqu'un prenne de ses nouvelles.",
        "de": "BAYMAX: {user} hat mich gebeten, dir Bescheid zu geben, dass jemand nach ihm/ihr sehen sollte.",
    },
    "all_clear": {
        "en": "BAYMAX: {user} has confirmed they are okay. The earlier alert is resolved.",
        "hi": "बेमैक्स: {user} ने पुष्टि की है कि वे ठीक हैं। पिछला अलर्ट सुलझ गया है।",
        "hi-Latn": "BAYMAX: {user} ne confirm kiya hai ki woh theek hain. Pichla alert resolve ho gaya hai.",
        "es": "BAYMAX: {user} ha confirmado que está bien. La alerta anterior está resuelta.",
        "fr": "BAYMAX : {user} a confirmé aller bien. L'alerte précédente est levée.",
        "de": "BAYMAX: {user} hat bestätigt, dass alles in Ordnung ist. Der vorherige Alarm ist aufgehoben.",
    },
    "test": {
        "en": "BAYMAX TEST: this is a test alert from {user}'s Baymax. No action is needed.",
        "hi": "बेमैक्स टेस्ट: यह {user} के बेमैक्स से एक टेस्ट अलर्ट है। कुछ करने की ज़रूरत नहीं है।",
        "hi-Latn": "BAYMAX TEST: yeh {user} ke Baymax se ek test alert hai. Kuch karne ki zaroorat nahi hai.",
        "es": "PRUEBA BAYMAX: esta es una alerta de prueba del Baymax de {user}. No hace falta hacer nada.",
        "fr": "TEST BAYMAX : ceci est une alerte de test du Baymax de {user}. Aucune action n'est nécessaire.",
        "de": "BAYMAX-TEST: Dies ist ein Testalarm von {user}s Baymax. Es ist nichts zu tun.",
    },
    "reason_prefix": {"en": "Reason", "hi": "कारण", "hi-Latn": "Kaaran", "es": "Motivo", "fr": "Motif", "de": "Grund"},
}

REASONS: dict[str, dict[str, str]] = {
    "fall_detected": {"en": "a possible fall was detected by the camera", "hi": "कैमरे ने संभावित गिरना पहचाना", "hi-Latn": "camera ne sambhavit girna detect kiya", "es": "la cámara detectó una posible caída", "fr": "la caméra a détecté une chute possible", "de": "die Kamera hat einen möglichen Sturz erkannt"},
    "fall_reported": {"en": "they reported a fall", "hi": "उन्होंने गिरने की बात बताई", "hi-Latn": "unhone girne ki baat batai", "es": "informó de una caída", "fr": "a signalé une chute", "de": "hat einen Sturz gemeldet"},
    "fall_cannot_get_up": {"en": "they fell and could not get up", "hi": "वे गिर गए और उठ नहीं पाए", "hi-Latn": "woh gir gaye aur uth nahi paaye", "es": "se cayó y no podía levantarse", "fr": "est tombé(e) et ne pouvait pas se relever", "de": "ist gestürzt und konnte nicht aufstehen"},
    "no_response": {"en": "no response to a check-in", "hi": "हालचाल पूछने पर कोई जवाब नहीं", "hi-Latn": "haal-chaal poochhne par koi jawab nahi", "es": "no respondió a una comprobación", "fr": "pas de réponse à une vérification", "de": "keine Antwort auf eine Nachfrage"},
    "critical": {"en": "they described emergency symptoms or asked for help", "hi": "उन्होंने आपातकालीन लक्षण बताए या मदद मांगी", "hi-Latn": "unhone emergency lakshan bataye ya madad maangi", "es": "describió síntomas de emergencia o pidió ayuda", "fr": "a décrit des symptômes d'urgence ou demandé de l'aide", "de": "hat Notfallsymptome beschrieben oder um Hilfe gebeten"},
    "high_pain": {"en": "they reported severe pain", "hi": "उन्होंने तेज़ दर्द बताया", "hi-Latn": "unhone tez dard bataya", "es": "informó de un dolor intenso", "fr": "a signalé une douleur intense", "de": "hat starke Schmerzen gemeldet"},
    "user_request": {"en": "they pressed the help button", "hi": "उन्होंने मदद का बटन दबाया", "hi-Latn": "unhone help button dabaya", "es": "pulsó el botón de ayuda", "fr": "a appuyé sur le bouton d'aide", "de": "hat den Hilfe-Knopf gedrückt"},
    "lying_inactive": {"en": "they were lying still for an unusually long time", "hi": "वे असामान्य रूप से लंबे समय तक बिना हिले लेटे रहे", "hi-Latn": "woh asaamanya roop se lambe samay tak bina hile lete rahe", "es": "estuvo tumbado sin moverse durante mucho tiempo", "fr": "est resté(e) allongé(e) immobile anormalement longtemps", "de": "hat ungewöhnlich lange regungslos gelegen"},
    "possible_emergency": {"en": "they said something was wrong", "hi": "उन्होंने कहा कि कुछ गड़बड़ है", "hi-Latn": "unhone kaha ki kuch gadbad hai", "es": "dijo que algo iba mal", "fr": "a dit que quelque chose n'allait pas", "de": "hat gesagt, dass etwas nicht stimmt"},
}


def render_alert(level: str, reason: str, lang: str, user: str, share_details: bool) -> str:
    lang = lang if lang in SUPPORTED_LANGS else "en"
    text = ALERT.get(level, ALERT["emergency"])[lang].format(user=user)
    if share_details and level not in ("all_clear", "test"):
        parts = [REASONS[r][lang] for r in (reason or "").split("+") if r in REASONS]
        if parts:
            text += f" {ALERT['reason_prefix'][lang]}: " + "; ".join(parts) + "."
    text += " " + time.strftime("%Y-%m-%d %H:%M")
    return text


class Dispatcher:
    def __init__(self, transport: Optional[httpx.BaseTransport] = None) -> None:
        self._transport = transport  # injectable for tests
        self.console_log: list[str] = []

    def dispatch(self, cfg: EscalationConfig, level: str, reason: str, contacts_consent: bool,
                 share_details: bool, force_send: bool = False) -> EscalationResult:
        """`force_send` bypasses dry-run; used only for the explicit 'send test alert' button."""
        if not contacts_consent:
            return EscalationResult(status="no_consent", level=level, reason=reason)
        if not cfg.contacts:
            return EscalationResult(status="no_contacts", level=level, reason=reason)
        if cfg.dry_run and not force_send:
            res = [ChannelResult(contact_id=c.id, channel=c.channel, ok=True,
                                 detail="dry-run: " + render_alert(level, reason, c.lang, cfg.user_display_name, share_details))
                   for c in cfg.contacts]
            return EscalationResult(status="dry_run", level=level, reason=reason, results=res)
        results = [self._send(c, cfg, render_alert(level, reason, c.lang, cfg.user_display_name, share_details))
                   for c in cfg.contacts]
        ok = sum(r.ok for r in results)
        status = "sent" if ok == len(results) else ("partial" if ok else "failed")
        return EscalationResult(status=status, level=level, reason=reason, results=results)

    def _send(self, c: Contact, cfg: EscalationConfig, text: str) -> ChannelResult:
        try:
            if c.channel == "console":
                self.console_log.append(f"[to {c.name}] {text}")
                print(f"[BAYMAX ESCALATION → {c.name}] {text}", flush=True)
                return ChannelResult(contact_id=c.id, channel=c.channel, ok=True, detail="printed")
            if c.channel == "webhook":
                with httpx.Client(timeout=cfg.webhook_timeout_s, transport=self._transport) as client:
                    r = client.post(c.address, json={"source": "baymax", "text": text, "contact": c.name},
                                    headers={"Title": "Baymax alert", "Priority": "urgent"})
                return ChannelResult(contact_id=c.id, channel=c.channel, ok=r.status_code < 300, detail=f"HTTP {r.status_code}")
            if c.channel == "email":
                host = os.environ.get("BAYMAX_SMTP_HOST")
                if not host:
                    return ChannelResult(contact_id=c.id, channel=c.channel, ok=False, detail="BAYMAX_SMTP_HOST not set")
                msg = EmailMessage()
                msg["Subject"] = "Baymax alert"
                msg["From"] = os.environ.get("BAYMAX_SMTP_FROM", "baymax@localhost")
                msg["To"] = c.address
                msg.set_content(text)
                with smtplib.SMTP(host, int(os.environ.get("BAYMAX_SMTP_PORT", "587")), timeout=10) as s:
                    if os.environ.get("BAYMAX_SMTP_STARTTLS", "1") == "1":
                        s.starttls()
                    if os.environ.get("BAYMAX_SMTP_USER"):
                        s.login(os.environ["BAYMAX_SMTP_USER"], os.environ.get("BAYMAX_SMTP_PASSWORD", ""))
                    s.send_message(msg)
                return ChannelResult(contact_id=c.id, channel=c.channel, ok=True, detail="sent")
        except Exception as e:  # a failed channel must never crash the safety path
            return ChannelResult(contact_id=c.id, channel=c.channel, ok=False, detail=f"{type(e).__name__}: {e}"[:200])
        return ChannelResult(contact_id=c.id, channel=c.channel, ok=False, detail="unknown channel")
