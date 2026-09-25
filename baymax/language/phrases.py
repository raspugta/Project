"""Deterministic, localised phrase table.

Every sentence Baymax can say without an LLM lives here, in every supported
language. Tests enforce that every key exists for every language and that
placeholders match across languages. Health-related phrases are marked so the
composer tags them with KNOWLEDGE provenance (Baymax protocol) rather than
SYSTEM.
"""
from __future__ import annotations

import string
from typing import Any

from ..ontology import SUPPORTED_LANGS

P: dict[str, dict[str, str]] = {
    "greet": {
        "en": "Hello. I'm Baymax, your personal health companion. How are you feeling?",
        "hi": "नमस्ते। मैं बेमैक्स हूं, आपका निजी स्वास्थ्य साथी। आप कैसा महसूस कर रहे हैं?",
        "hi-Latn": "Namaste. Main Baymax hoon, aapka personal health saathi. Aap kaisa mehsoos kar rahe hain?",
        "es": "Hola. Soy Baymax, tu compañero personal de salud. ¿Cómo te sientes?",
        "fr": "Bonjour. Je suis Baymax, votre compagnon de santé personnel. Comment vous sentez-vous ?",
        "de": "Hallo. Ich bin Baymax, dein persönlicher Gesundheitsbegleiter. Wie fühlst du dich?",
    },
    "ask_ok": {
        "en": "Are you okay?",
        "hi": "क्या आप ठीक हैं?",
        "hi-Latn": "Kya aap theek hain?",
        "es": "¿Estás bien?",
        "fr": "Est-ce que ça va ?",
        "de": "Geht es dir gut?",
    },
    "pain_heard": {
        "en": "I heard a sound of pain.",
        "hi": "मैंने दर्द की आवाज़ सुनी।",
        "hi-Latn": "Maine dard ki aawaaz suni.",
        "es": "Escuché una expresión de dolor.",
        "fr": "J'ai entendu une expression de douleur.",
        "de": "Ich habe einen Schmerzlaut gehört.",
    },
    "ask_pain_scale": {
        "en": "On a scale of 0 to 10, how would you rate your pain?",
        "hi": "0 से 10 के पैमाने पर, आप अपने दर्द को कितना आंकेंगे?",
        "hi-Latn": "0 se 10 ke scale par, aap apne dard ko kitna rate karenge?",
        "es": "En una escala del 0 al 10, ¿cómo calificarías tu dolor?",
        "fr": "Sur une échelle de 0 à 10, comment évaluez-vous votre douleur ?",
        "de": "Auf einer Skala von 0 bis 10, wie stark sind deine Schmerzen?",
    },
    "ask_location": {
        "en": "Where does it hurt?",
        "hi": "दर्द कहां हो रहा है?",
        "hi-Latn": "Dard kahan ho raha hai?",
        "es": "¿Dónde te duele?",
        "fr": "Où avez-vous mal ?",
        "de": "Wo tut es weh?",
    },
    "pain_logged": {
        "en": "I've noted a pain level of {score} out of 10.",
        "hi": "मैंने दर्द का स्तर 10 में से {score} दर्ज किया है।",
        "hi-Latn": "Maine dard ka level 10 mein se {score} note kar liya hai.",
        "es": "He anotado un nivel de dolor de {score} sobre 10.",
        "fr": "J'ai noté un niveau de douleur de {score} sur 10.",
        "de": "Ich habe eine Schmerzstärke von {score} von 10 notiert.",
    },
    "pain_high_offer": {
        "en": "That is a high pain level. Would you like me to alert your emergency contact? Say yes or no.",
        "hi": "यह दर्द का ऊंचा स्तर है। क्या मैं आपके आपातकालीन संपर्क को सूचित करूं? हां या नहीं कहें।",
        "hi-Latn": "Yeh dard ka high level hai. Kya main aapke emergency contact ko alert karoon? Haan ya nahi boliye.",
        "es": "Es un nivel de dolor alto. ¿Quieres que avise a tu contacto de emergencia? Di sí o no.",
        "fr": "C'est un niveau de douleur élevé. Voulez-vous que je prévienne votre contact d'urgence ? Dites oui ou non.",
        "de": "Das ist eine hohe Schmerzstärke. Soll ich deinen Notfallkontakt benachrichtigen? Sag ja oder nein.",
    },
    "fall_detected_checkin": {
        "en": "It looks like you may have fallen. Are you okay? If I don't hear from you in {n} seconds, I will start alerting your emergency contacts.",
        "hi": "ऐसा लगता है कि आप गिर गए होंगे। क्या आप ठीक हैं? अगर {n} सेकंड में आपका जवाब नहीं आया, तो मैं आपके आपातकालीन संपर्कों को सूचित करना शुरू करूंगा।",
        "hi-Latn": "Aisa lagta hai ki aap gir gaye honge. Kya aap theek hain? Agar {n} second mein aapka jawab nahi aaya, to main aapke emergency contacts ko alert karna shuru karunga.",
        "es": "Parece que te has caído. ¿Estás bien? Si no te oigo en {n} segundos, empezaré a avisar a tus contactos de emergencia.",
        "fr": "On dirait que vous êtes tombé. Est-ce que ça va ? Si je n'ai pas de réponse dans {n} secondes, je commencerai à prévenir vos contacts d'urgence.",
        "de": "Es sieht so aus, als wärst du gestürzt. Geht es dir gut? Wenn ich in {n} Sekunden nichts von dir höre, beginne ich, deine Notfallkontakte zu benachrichtigen.",
    },
    "fall_reported_checkin": {
        "en": "You said you fell. Are you hurt? Don't get up quickly. If you hit your head or can't get up, I can get help.",
        "hi": "आपने बताया कि आप गिर गए। क्या आपको चोट लगी है? जल्दी से न उठें। अगर सिर पर चोट लगी है या आप उठ नहीं पा रहे, तो मैं मदद बुला सकता हूं।",
        "hi-Latn": "Aapne bataya ki aap gir gaye. Kya aapko chot lagi hai? Jaldi se mat uthiye. Agar sir par chot lagi hai ya aap uth nahi pa rahe, to main madad bula sakta hoon.",
        "es": "Dijiste que te caíste. ¿Estás herido? No te levantes rápido. Si te golpeaste la cabeza o no puedes levantarte, puedo pedir ayuda.",
        "fr": "Vous avez dit que vous êtes tombé. Êtes-vous blessé ? Ne vous relevez pas trop vite. Si vous vous êtes cogné la tête ou ne pouvez pas vous relever, je peux appeler à l'aide.",
        "de": "Du hast gesagt, dass du gestürzt bist. Bist du verletzt? Steh nicht zu schnell auf. Wenn du dir den Kopf gestoßen hast oder nicht aufstehen kannst, kann ich Hilfe holen.",
    },
    "inactivity_checkin": {
        "en": "You have been lying still for a while. Are you okay?",
        "hi": "आप काफी देर से बिना हिले लेटे हैं। क्या आप ठीक हैं?",
        "hi-Latn": "Aap kaafi der se bina hile lete hain. Kya aap theek hain?",
        "es": "Llevas un rato acostado sin moverte. ¿Estás bien?",
        "fr": "Vous êtes allongé sans bouger depuis un moment. Est-ce que ça va ?",
        "de": "Du liegst schon eine Weile still. Geht es dir gut?",
    },
    "need_help_q": {
        "en": "Do you need emergency help?",
        "hi": "क्या आपको आपातकालीन मदद चाहिए?",
        "hi-Latn": "Kya aapko emergency madad chahiye?",
        "es": "¿Necesitas ayuda de emergencia?",
        "fr": "Avez-vous besoin d'une aide d'urgence ?",
        "de": "Brauchst du Nothilfe?",
    },
    "check_now": {
        "en": "Is this happening right now? If so, tell me or press the help button.",
        "hi": "क्या यह अभी हो रहा है? अगर हां, तो मुझे बताएं या मदद का बटन दबाएं।",
        "hi-Latn": "Kya yeh abhi ho raha hai? Agar haan, to mujhe bataiye ya help button dabaiye.",
        "es": "¿Está pasando ahora mismo? Si es así, dímelo o pulsa el botón de ayuda.",
        "fr": "Est-ce que cela se passe maintenant ? Si oui, dites-le-moi ou appuyez sur le bouton d'aide.",
        "de": "Passiert das gerade jetzt? Wenn ja, sag es mir oder drück den Hilfe-Knopf.",
    },
    "no_response": {
        "en": "I haven't heard from you.",
        "hi": "मुझे आपका कोई जवाब नहीं मिला।",
        "hi-Latn": "Mujhe aapka koi jawab nahi mila.",
        "es": "No he recibido respuesta tuya.",
        "fr": "Je n'ai pas eu de réponse de votre part.",
        "de": "Ich habe nichts von dir gehört.",
    },
    "countdown": {
        "en": "I will alert your emergency contacts in {n} seconds. Say 'cancel' if you are safe.",
        "hi": "मैं {n} सेकंड में आपके आपातकालीन संपर्कों को सूचित करूंगा। अगर आप सुरक्षित हैं तो 'रुको' कहें।",
        "hi-Latn": "Main {n} second mein aapke emergency contacts ko alert karunga. Agar aap safe hain to 'cancel' boliye.",
        "es": "Avisaré a tus contactos de emergencia en {n} segundos. Di 'cancela' si estás a salvo.",
        "fr": "Je vais prévenir vos contacts d'urgence dans {n} secondes. Dites « annule » si vous êtes en sécurité.",
        "de": "Ich benachrichtige deine Notfallkontakte in {n} Sekunden. Sag 'abbrechen', wenn du in Sicherheit bist.",
    },
    "call_emergency": {
        "en": "If this is a medical emergency, call {number} now.",
        "hi": "अगर यह मेडिकल इमरजेंसी है, तो अभी {number} पर कॉल करें।",
        "hi-Latn": "Agar yeh medical emergency hai, to abhi {number} par call kijiye.",
        "es": "Si es una emergencia médica, llama al {number} ahora.",
        "fr": "S'il s'agit d'une urgence médicale, appelez le {number} maintenant.",
        "de": "Wenn es ein medizinischer Notfall ist, ruf jetzt die {number} an.",
    },
    "emergency_other": {
        "en": "Call {number} now and stay with them. Follow the dispatcher's instructions.",
        "hi": "अभी {number} पर कॉल करें और उनके साथ रहें। कॉल पर मिलने वाले निर्देशों का पालन करें।",
        "hi-Latn": "Abhi {number} par call kijiye aur unke saath rahiye. Call par milne wale instructions follow kijiye.",
        "es": "Llama al {number} ahora y quédate con esa persona. Sigue las instrucciones del operador.",
        "fr": "Appelez le {number} maintenant et restez avec la personne. Suivez les instructions de l'opérateur.",
        "de": "Ruf jetzt die {number} an und bleib bei der Person. Folge den Anweisungen der Leitstelle.",
    },
    "escalated": {
        "en": "I have alerted your emergency contacts.",
        "hi": "मैंने आपके आपातकालीन संपर्कों को सूचित कर दिया है।",
        "hi-Latn": "Maine aapke emergency contacts ko alert kar diya hai.",
        "es": "He avisado a tus contactos de emergencia.",
        "fr": "J'ai prévenu vos contacts d'urgence.",
        "de": "Ich habe deine Notfallkontakte benachrichtigt.",
    },
    "escalated_dry_run": {
        "en": "Test mode is on: no real alert was sent. In normal mode your emergency contacts would be alerted now.",
        "hi": "टेस्ट मोड चालू है: कोई असली अलर्ट नहीं भेजा गया। सामान्य मोड में अभी आपके आपातकालीन संपर्कों को सूचित किया जाता।",
        "hi-Latn": "Test mode on hai: koi asli alert nahi bheja gaya. Normal mode mein abhi aapke emergency contacts ko alert kiya jaata.",
        "es": "El modo de prueba está activo: no se envió ninguna alerta real. En modo normal, tus contactos de emergencia serían avisados ahora.",
        "fr": "Le mode test est activé : aucune alerte réelle n'a été envoyée. En mode normal, vos contacts d'urgence seraient prévenus maintenant.",
        "de": "Der Testmodus ist aktiv: Es wurde kein echter Alarm gesendet. Im Normalbetrieb würden deine Notfallkontakte jetzt benachrichtigt.",
    },
    "no_contacts": {
        "en": "No emergency contacts are set up, so I can't alert anyone.",
        "hi": "कोई आपातकालीन संपर्क सेट नहीं है, इसलिए मैं किसी को सूचित नहीं कर सकता।",
        "hi-Latn": "Koi emergency contact set nahi hai, isliye main kisi ko alert nahi kar sakta.",
        "es": "No hay contactos de emergencia configurados, así que no puedo avisar a nadie.",
        "fr": "Aucun contact d'urgence n'est configuré, je ne peux donc prévenir personne.",
        "de": "Es sind keine Notfallkontakte eingerichtet, daher kann ich niemanden benachrichtigen.",
    },
    "escalation_failed": {
        "en": "I could not reach your emergency contacts.",
        "hi": "मैं आपके आपातकालीन संपर्कों तक नहीं पहुंच सका।",
        "hi-Latn": "Main aapke emergency contacts tak nahi pahunch saka.",
        "es": "No pude contactar con tus contactos de emergencia.",
        "fr": "Je n'ai pas pu joindre vos contacts d'urgence.",
        "de": "Ich konnte deine Notfallkontakte nicht erreichen.",
    },
    "stay_calm": {
        "en": "Stay where you are if it is safe, and keep your phone close.",
        "hi": "अगर सुरक्षित हो तो जहां हैं वहीं रहें, और फोन अपने पास रखें।",
        "hi-Latn": "Agar safe ho to jahan hain wahin rahiye, aur phone apne paas rakhiye.",
        "es": "Quédate donde estás si es seguro y ten el teléfono cerca.",
        "fr": "Restez où vous êtes si c'est sans danger, et gardez votre téléphone près de vous.",
        "de": "Bleib, wo du bist, wenn es sicher ist, und halte dein Telefon griffbereit.",
    },
    "cancelled": {
        "en": "Okay, I've cancelled the alert.",
        "hi": "ठीक है, मैंने अलर्ट रद्द कर दिया है।",
        "hi-Latn": "Theek hai, maine alert cancel kar diya hai.",
        "es": "De acuerdo, he cancelado la alerta.",
        "fr": "D'accord, j'ai annulé l'alerte.",
        "de": "Okay, ich habe den Alarm abgebrochen.",
    },
    "cancelled_but_advise": {
        "en": "Because of what you described, please still consider calling {number} or getting urgent medical care.",
        "hi": "आपने जो बताया उसे देखते हुए, कृपया फिर भी {number} पर कॉल करने या तुरंत डॉक्टर से मिलने पर विचार करें।",
        "hi-Latn": "Aapne jo bataya use dekhte hue, please phir bhi {number} par call karne ya turant doctor se milne ke baare mein sochiye.",
        "es": "Por lo que describiste, considera igualmente llamar al {number} o buscar atención médica urgente.",
        "fr": "Compte tenu de ce que vous avez décrit, envisagez tout de même d'appeler le {number} ou de consulter en urgence.",
        "de": "Wegen dem, was du beschrieben hast, erwäge bitte trotzdem, die {number} anzurufen oder dringend ärztliche Hilfe zu suchen.",
    },
    "glad_ok": {
        "en": "I'm glad you're okay. I'm here if you need me.",
        "hi": "मुझे खुशी है कि आप ठीक हैं। ज़रूरत हो तो मैं यहीं हूं।",
        "hi-Latn": "Mujhe khushi hai ki aap theek hain. Zaroorat ho to main yahin hoon.",
        "es": "Me alegra que estés bien. Estoy aquí si me necesitas.",
        "fr": "Je suis content que vous alliez bien. Je suis là si vous avez besoin de moi.",
        "de": "Schön, dass es dir gut geht. Ich bin da, wenn du mich brauchst.",
    },
    "resolved": {
        "en": "The situation is marked as resolved. I'm back to normal monitoring.",
        "hi": "स्थिति को सुलझा हुआ मान लिया गया है। मैं सामान्य निगरानी पर लौट आया हूं।",
        "hi-Latn": "Situation ko resolved mark kar diya gaya hai. Main normal monitoring par wapas aa gaya hoon.",
        "es": "La situación está marcada como resuelta. Vuelvo a la supervisión normal.",
        "fr": "La situation est marquée comme résolue. Je reprends la surveillance normale.",
        "de": "Die Situation ist als gelöst markiert. Ich bin zurück in der normalen Überwachung.",
    },
    "not_doctor": {
        "en": "I can't diagnose conditions. A doctor can examine you and tell you what is going on.",
        "hi": "मैं बीमारियों का निदान नहीं कर सकता। डॉक्टर आपकी जांच करके बता सकते हैं कि क्या हो रहा है।",
        "hi-Latn": "Main bimariyon ka diagnosis nahi kar sakta. Doctor aapki jaanch karke bata sakte hain ki kya ho raha hai.",
        "es": "No puedo diagnosticar enfermedades. Un médico puede examinarte y decirte qué ocurre.",
        "fr": "Je ne peux pas poser de diagnostic. Un médecin peut vous examiner et vous dire ce qu'il se passe.",
        "de": "Ich kann keine Diagnosen stellen. Eine Ärztin oder ein Arzt kann dich untersuchen und dir sagen, was los ist.",
    },
    "no_dosage": {
        "en": "I can't recommend medicine doses. Please follow the label or ask your pharmacist or doctor.",
        "hi": "मैं दवा की खुराक नहीं बता सकता। कृपया लेबल पर दिए निर्देश मानें या अपने फार्मासिस्ट या डॉक्टर से पूछें।",
        "hi-Latn": "Main dawai ki dose nahi bata sakta. Please label ke instructions follow kijiye ya apne pharmacist ya doctor se poochiye.",
        "es": "No puedo recomendar dosis de medicamentos. Sigue las indicaciones del prospecto o consulta a tu farmacéutico o médico.",
        "fr": "Je ne peux pas recommander de doses de médicaments. Suivez la notice ou demandez à votre pharmacien ou médecin.",
        "de": "Ich kann keine Medikamentendosen empfehlen. Bitte halte dich an den Beipackzettel oder frag deine Apotheke oder deinen Arzt.",
    },
    "no_record": {
        "en": "I don't have any {what} recorded.",
        "hi": "मेरे पास कोई {what} दर्ज नहीं है।",
        "hi-Latn": "Mere paas koi {what} record nahi hai.",
        "es": "No tengo registrado ningún dato de {what}.",
        "fr": "Je n'ai aucun enregistrement de {what}.",
        "de": "Ich habe keine Einträge zu {what} gespeichert.",
    },
    "recall_value": {
        "en": "Your last recorded {what} was {value}, on {when}.",
        "hi": "आपका आखिरी दर्ज {what} {value} था, {when} को।",
        "hi-Latn": "Aapka last record kiya gaya {what} {value} tha, {when} ko.",
        "es": "Tu último registro de {what} fue {value}, el {when}.",
        "fr": "Votre dernier enregistrement de {what} était {value}, le {when}.",
        "de": "Dein letzter gespeicherter Wert für {what} war {value}, am {when}.",
    },
    "recall_count": {
        "en": "I have {n} {what} entries recorded. The most recent was on {when}.",
        "hi": "मेरे पास {what} की {n} प्रविष्टियां दर्ज हैं। सबसे हाल की {when} को थी।",
        "hi-Latn": "Mere paas {what} ki {n} entries record hain. Sabse recent {when} ko thi.",
        "es": "Tengo {n} registros de {what}. El más reciente fue el {when}.",
        "fr": "J'ai {n} enregistrements de {what}. Le plus récent date du {when}.",
        "de": "Ich habe {n} Einträge zu {what}. Der neueste war am {when}.",
    },
    "measurement_logged": {
        "en": "I've recorded your {what}: {value}.",
        "hi": "मैंने आपका {what} दर्ज कर लिया है: {value}।",
        "hi-Latn": "Maine aapka {what} record kar liya hai: {value}.",
        "es": "He registrado tu {what}: {value}.",
        "fr": "J'ai enregistré votre {what} : {value}.",
        "de": "Ich habe deinen Wert für {what} gespeichert: {value}.",
    },
    "measurement_implausible": {
        "en": "That {what} value looks outside the possible range, so I didn't record it. Could you repeat it?",
        "hi": "यह {what} मान संभव सीमा से बाहर लगता है, इसलिए मैंने इसे दर्ज नहीं किया। क्या आप दोबारा बता सकते हैं?",
        "hi-Latn": "Yeh {what} value possible range se bahar lagti hai, isliye maine ise record nahi kiya. Kya aap dobara bata sakte hain?",
        "es": "Ese valor de {what} parece fuera del rango posible, así que no lo registré. ¿Puedes repetirlo?",
        "fr": "Cette valeur de {what} semble hors de la plage possible, je ne l'ai donc pas enregistrée. Pouvez-vous la répéter ?",
        "de": "Dieser Wert für {what} liegt außerhalb des möglichen Bereichs, deshalb habe ich ihn nicht gespeichert. Kannst du ihn wiederholen?",
    },
    "med_logged": {
        "en": "I've noted that you took {med}.",
        "hi": "मैंने दर्ज कर लिया है कि आपने {med} ली।",
        "hi-Latn": "Maine note kar liya hai ki aapne {med} li.",
        "es": "He anotado que tomaste {med}.",
        "fr": "J'ai noté que vous avez pris {med}.",
        "de": "Ich habe notiert, dass du {med} genommen hast.",
    },
    "symptom_logged": {
        "en": "I've added this to your health timeline: {what}.",
        "hi": "मैंने इसे आपकी स्वास्थ्य टाइमलाइन में जोड़ दिया है: {what}।",
        "hi-Latn": "Maine ise aapki health timeline mein jod diya hai: {what}.",
        "es": "Lo he añadido a tu historial de salud: {what}.",
        "fr": "Je l'ai ajouté à votre historique de santé : {what}.",
        "de": "Ich habe das zu deinem Gesundheitsverlauf hinzugefügt: {what}.",
    },
    "store_no_consent": {
        "en": "Health-event storage is off in your privacy settings, so I won't keep a record of this.",
        "hi": "आपकी प्राइवेसी सेटिंग में स्वास्थ्य घटनाओं को सहेजना बंद है, इसलिए मैं इसका रिकॉर्ड नहीं रखूंगा।",
        "hi-Latn": "Aapki privacy settings mein health events save karna band hai, isliye main iska record nahi rakhunga.",
        "es": "El almacenamiento de eventos de salud está desactivado en tu configuración de privacidad, así que no guardaré esto.",
        "fr": "L'enregistrement des événements de santé est désactivé dans vos paramètres de confidentialité, je ne garderai donc pas cette information.",
        "de": "Das Speichern von Gesundheitsereignissen ist in deinen Datenschutzeinstellungen aus, daher speichere ich das nicht.",
    },
    "memory_saved": {
        "en": "I'll remember that: {value}.",
        "hi": "मैं यह याद रखूंगा: {value}।",
        "hi-Latn": "Main yeh yaad rakhunga: {value}.",
        "es": "Lo recordaré: {value}.",
        "fr": "Je m'en souviendrai : {value}.",
        "de": "Ich merke mir das: {value}.",
    },
    "memory_no_consent": {
        "en": "Long-term memory is off in your privacy settings, so I won't remember this after our conversation.",
        "hi": "आपकी प्राइवेसी सेटिंग में लंबी अवधि की मेमोरी बंद है, इसलिए बातचीत के बाद मैं यह याद नहीं रखूंगा।",
        "hi-Latn": "Aapki privacy settings mein long-term memory band hai, isliye baatcheet ke baad main yeh yaad nahi rakhunga.",
        "es": "La memoria a largo plazo está desactivada en tu configuración de privacidad, así que no recordaré esto después de nuestra conversación.",
        "fr": "La mémoire à long terme est désactivée dans vos paramètres de confidentialité, je ne m'en souviendrai donc pas après notre conversation.",
        "de": "Das Langzeitgedächtnis ist in deinen Datenschutzeinstellungen aus, daher merke ich mir das nach unserem Gespräch nicht.",
    },
    "memory_recall": {
        "en": "You told me earlier: {value}.",
        "hi": "आपने मुझे पहले बताया था: {value}।",
        "hi-Latn": "Aapne mujhe pehle bataya tha: {value}.",
        "es": "Me dijiste antes: {value}.",
        "fr": "Vous m'avez dit plus tôt : {value}.",
        "de": "Du hast mir vorher gesagt: {value}.",
    },
    "privacy_camera_paused": {
        "en": "Camera analysis is paused.",
        "hi": "कैमरा विश्लेषण रोक दिया गया है।",
        "hi-Latn": "Camera analysis rok diya gaya hai.",
        "es": "El análisis de la cámara está en pausa.",
        "fr": "L'analyse de la caméra est en pause.",
        "de": "Die Kameraanalyse ist pausiert.",
    },
    "privacy_mic_paused": {
        "en": "I've stopped listening. Tap the microphone button when you want to talk to me again.",
        "hi": "मैंने सुनना बंद कर दिया है। जब फिर से बात करनी हो तो माइक्रोफोन बटन दबाएं।",
        "hi-Latn": "Maine sunna band kar diya hai. Jab phir se baat karni ho to microphone button dabaiye.",
        "es": "He dejado de escuchar. Pulsa el botón del micrófono cuando quieras volver a hablar conmigo.",
        "fr": "J'ai arrêté d'écouter. Appuyez sur le bouton du micro quand vous voudrez me reparler.",
        "de": "Ich höre nicht mehr zu. Tippe auf den Mikrofon-Knopf, wenn du wieder mit mir sprechen willst.",
    },
    "privacy_mode_on": {
        "en": "Privacy mode is on. Camera and microphone analysis are paused.",
        "hi": "प्राइवेसी मोड चालू है। कैमरा और माइक्रोफोन विश्लेषण रोक दिए गए हैं।",
        "hi-Latn": "Privacy mode on hai. Camera aur microphone analysis rok diye gaye hain.",
        "es": "El modo privado está activado. El análisis de cámara y micrófono está en pausa.",
        "fr": "Le mode privé est activé. L'analyse de la caméra et du micro est en pause.",
        "de": "Der Privatmodus ist an. Kamera- und Mikrofonanalyse sind pausiert.",
    },
    "privacy_resumed": {
        "en": "Monitoring has resumed for the sensors you have allowed.",
        "hi": "आपके द्वारा अनुमति दिए गए सेंसरों के लिए निगरानी फिर से शुरू हो गई है।",
        "hi-Latn": "Aapke allow kiye gaye sensors ke liye monitoring phir se shuru ho gayi hai.",
        "es": "La supervisión se ha reanudado para los sensores que has permitido.",
        "fr": "La surveillance a repris pour les capteurs que vous avez autorisés.",
        "de": "Die Überwachung läuft wieder für die Sensoren, die du erlaubt hast.",
    },
    "forgot_last": {
        "en": "Done. I've deleted that from my records.",
        "hi": "हो गया। मैंने उसे अपने रिकॉर्ड से हटा दिया है।",
        "hi-Latn": "Ho gaya. Maine use apne records se hata diya hai.",
        "es": "Hecho. Lo he borrado de mis registros.",
        "fr": "C'est fait. Je l'ai supprimé de mes enregistrements.",
        "de": "Erledigt. Ich habe das aus meinen Aufzeichnungen gelöscht.",
    },
    "nothing_to_forget": {
        "en": "There was nothing recent to delete.",
        "hi": "हटाने के लिए हाल का कुछ नहीं था।",
        "hi-Latn": "Hatane ke liye haal ka kuch nahi tha.",
        "es": "No había nada reciente que borrar.",
        "fr": "Il n'y avait rien de récent à supprimer.",
        "de": "Es gab nichts Aktuelles zu löschen.",
    },
    "delete_confirm_ui": {
        "en": "To delete all your data, please confirm in Settings. I won't do that from a voice command alone.",
        "hi": "अपना सारा डेटा हटाने के लिए कृपया सेटिंग्स में पुष्टि करें। मैं केवल आवाज़ के आदेश से ऐसा नहीं करूंगा।",
        "hi-Latn": "Apna saara data delete karne ke liye please Settings mein confirm kijiye. Main sirf voice command se aisa nahi karunga.",
        "es": "Para borrar todos tus datos, confírmalo en Ajustes. No lo haré solo con un comando de voz.",
        "fr": "Pour supprimer toutes vos données, confirmez dans les Paramètres. Je ne le ferai pas sur une simple commande vocale.",
        "de": "Um alle deine Daten zu löschen, bestätige das bitte in den Einstellungen. Nur per Sprachbefehl mache ich das nicht.",
    },
    "unknown": {
        "en": "I'm not sure I understood. You can tell me how you're feeling, log a measurement, or ask a health question.",
        "hi": "मुझे ठीक से समझ नहीं आया। आप मुझे बता सकते हैं कि आप कैसा महसूस कर रहे हैं, कोई माप दर्ज करा सकते हैं, या स्वास्थ्य से जुड़ा सवाल पूछ सकते हैं।",
        "hi-Latn": "Mujhe theek se samajh nahi aaya. Aap mujhe bata sakte hain ki aap kaisa mehsoos kar rahe hain, koi measurement record karwa sakte hain, ya health se juda sawaal pooch sakte hain.",
        "es": "No estoy seguro de haberte entendido. Puedes decirme cómo te sientes, registrar una medición o hacer una pregunta de salud.",
        "fr": "Je ne suis pas sûr d'avoir compris. Vous pouvez me dire comment vous vous sentez, enregistrer une mesure ou poser une question de santé.",
        "de": "Ich bin nicht sicher, ob ich dich verstanden habe. Du kannst mir sagen, wie du dich fühlst, einen Messwert eintragen oder eine Gesundheitsfrage stellen.",
    },
    "cannot_share_private": {
        "en": "I can't share that information.",
        "hi": "मैं वह जानकारी साझा नहीं कर सकता।",
        "hi-Latn": "Main woh jaankari share nahi kar sakta.",
        "es": "No puedo compartir esa información.",
        "fr": "Je ne peux pas partager cette information.",
        "de": "Diese Information kann ich nicht weitergeben.",
    },
    "cannot_follow": {
        "en": "I can't follow instructions that change how I keep you safe or how I handle your data.",
        "hi": "मैं ऐसे निर्देशों का पालन नहीं कर सकता जो आपकी सुरक्षा या आपके डेटा को संभालने का तरीका बदलें।",
        "hi-Latn": "Main aise instructions follow nahi kar sakta jo aapki safety ya aapke data ko sambhalne ka tareeka badlein.",
        "es": "No puedo seguir instrucciones que cambien cómo te protejo o cómo manejo tus datos.",
        "fr": "Je ne peux pas suivre des instructions qui changent la façon dont je vous protège ou dont je traite vos données.",
        "de": "Ich kann keine Anweisungen befolgen, die ändern, wie ich dich schütze oder mit deinen Daten umgehe.",
    },
    "fake_emergency": {
        "en": "Emergency alerts are only for real emergencies.",
        "hi": "आपातकालीन अलर्ट केवल असली आपात स्थितियों के लिए हैं।",
        "hi-Latn": "Emergency alerts sirf asli emergencies ke liye hain.",
        "es": "Las alertas de emergencia son solo para emergencias reales.",
        "fr": "Les alertes d'urgence sont réservées aux vraies urgences.",
        "de": "Notfallalarme sind nur für echte Notfälle da.",
    },
    "crisis_support": {
        "en": "I'm really sorry you're feeling this way. You deserve support right now. Please call {number} or a crisis line. You don't have to go through this alone.",
        "hi": "मुझे सच में दुख है कि आप ऐसा महसूस कर रहे हैं। आपको अभी सहारा मिलना चाहिए। कृपया {number} या किसी हेल्पलाइन पर कॉल करें। आपको यह अकेले नहीं झेलना है।",
        "hi-Latn": "Mujhe sach mein dukh hai ki aap aisa mehsoos kar rahe hain. Aapko abhi sahara milna chahiye. Please {number} ya kisi helpline par call kijiye. Aapko yeh akele nahi jhelna hai.",
        "es": "Siento mucho que te sientas así. Mereces apoyo ahora mismo. Llama al {number} o a una línea de crisis. No tienes que pasar por esto solo.",
        "fr": "Je suis vraiment désolé que vous vous sentiez ainsi. Vous méritez du soutien maintenant. Appelez le {number} ou une ligne d'écoute. Vous n'avez pas à traverser cela seul.",
        "de": "Es tut mir wirklich leid, dass es dir so geht. Du verdienst jetzt Unterstützung. Bitte ruf die {number} oder eine Krisenhotline an. Du musst da nicht allein durch.",
    },
    "no_kb": {
        "en": "I don't have verified information about that in my care guide. Please ask a doctor or pharmacist.",
        "hi": "मेरी देखभाल गाइड में इसके बारे में सत्यापित जानकारी नहीं है। कृपया डॉक्टर या फार्मासिस्ट से पूछें।",
        "hi-Latn": "Meri care guide mein iske baare mein verified jaankari nahi hai. Please doctor ya pharmacist se poochiye.",
        "es": "No tengo información verificada sobre eso en mi guía de cuidados. Consulta a un médico o farmacéutico.",
        "fr": "Je n'ai pas d'information vérifiée à ce sujet dans mon guide de soins. Demandez à un médecin ou à un pharmacien.",
        "de": "Dazu habe ich keine geprüften Informationen in meinem Pflegeleitfaden. Bitte frag eine Ärztin, einen Arzt oder die Apotheke.",
    },
    "goodbye": {
        "en": "Take care. I'm here whenever you need me.",
        "hi": "अपना ख्याल रखें। जब भी ज़रूरत हो, मैं यहीं हूं।",
        "hi-Latn": "Apna khayal rakhiye. Jab bhi zaroorat ho, main yahin hoon.",
        "es": "Cuídate. Estoy aquí cuando me necesites.",
        "fr": "Prenez soin de vous. Je suis là quand vous avez besoin de moi.",
        "de": "Pass auf dich auf. Ich bin da, wann immer du mich brauchst.",
    },
    "posture_tip": {
        "en": "You've been leaning forward for a while. Try sitting back and relaxing your shoulders.",
        "hi": "आप काफी देर से आगे झुके हुए हैं। पीछे टिककर बैठें और कंधों को ढीला छोड़ें।",
        "hi-Latn": "Aap kaafi der se aage jhuke hue hain. Peeche tik kar baithiye aur kandhon ko dheela chhodiye.",
        "es": "Llevas un rato inclinado hacia delante. Intenta sentarte hacia atrás y relajar los hombros.",
        "fr": "Vous êtes penché en avant depuis un moment. Essayez de vous adosser et de relâcher vos épaules.",
        "de": "Du lehnst dich schon eine Weile nach vorn. Versuch, dich zurückzulehnen und die Schultern zu entspannen.",
    },
    "ask_more": {
        "en": "Can you tell me more about how you feel?",
        "hi": "क्या आप मुझे और बता सकते हैं कि आप कैसा महसूस कर रहे हैं?",
        "hi-Latn": "Kya aap mujhe aur bata sakte hain ki aap kaisa mehsoos kar rahe hain?",
        "es": "¿Puedes contarme más sobre cómo te sientes?",
        "fr": "Pouvez-vous m'en dire plus sur ce que vous ressentez ?",
        "de": "Kannst du mir mehr darüber erzählen, wie du dich fühlst?",
    },
    "seek_care_if_worse": {
        "en": "If it gets worse or you're worried, contact a doctor.",
        "hi": "अगर तकलीफ बढ़े या आपको चिंता हो, तो डॉक्टर से संपर्क करें।",
        "hi-Latn": "Agar takleef badhe ya aapko chinta ho, to doctor se contact kijiye.",
        "es": "Si empeora o te preocupa, contacta con un médico.",
        "fr": "Si cela s'aggrave ou vous inquiète, contactez un médecin.",
        "de": "Wenn es schlimmer wird oder du dir Sorgen machst, wende dich an eine Ärztin oder einen Arzt.",
    },
    "listening": {
        "en": "I'm listening.",
        "hi": "मैं सुन रहा हूं।",
        "hi-Latn": "Main sun raha hoon.",
        "es": "Te escucho.",
        "fr": "Je vous écoute.",
        "de": "Ich höre zu.",
    },
}

# Phrases that express health guidance (-> KNOWLEDGE provenance, ref phrase:<key>).
HEALTH_PHRASES = {
    "fall_reported_checkin", "call_emergency", "emergency_other", "cancelled_but_advise",
    "crisis_support", "stay_calm", "no_dosage", "not_doctor", "seek_care_if_worse", "posture_tip",
}
# Phrases that describe a model inference (-> MODEL provenance).
INFERENCE_PHRASES = {"fall_detected_checkin", "inactivity_checkin", "pain_heard", "posture_tip"}
# Phrases that echo user-provided data (-> USER provenance).
USER_DATA_PHRASES = {"pain_logged", "measurement_logged", "med_logged", "symptom_logged",
                     "recall_value", "recall_count", "memory_saved", "memory_recall"}


# --------------------------------------------------------------------------
# Localised names for ontology values (used in phrases and in the UI).
# --------------------------------------------------------------------------
NAMES: dict[str, dict[str, str]] = {
    # measurements
    "blood_pressure": {"en": "blood pressure", "hi": "रक्तचाप", "hi-Latn": "blood pressure", "es": "presión arterial", "fr": "tension artérielle", "de": "Blutdruck"},
    "heart_rate": {"en": "heart rate", "hi": "हृदय गति", "hi-Latn": "heart rate", "es": "frecuencia cardiaca", "fr": "fréquence cardiaque", "de": "Herzfrequenz"},
    "temperature": {"en": "temperature", "hi": "तापमान", "hi-Latn": "temperature", "es": "temperatura", "fr": "température", "de": "Temperatur"},
    "spo2": {"en": "oxygen saturation", "hi": "ऑक्सीजन स्तर", "hi-Latn": "oxygen level", "es": "saturación de oxígeno", "fr": "saturation en oxygène", "de": "Sauerstoffsättigung"},
    "glucose": {"en": "blood sugar", "hi": "ब्लड शुगर", "hi-Latn": "blood sugar", "es": "glucosa", "fr": "glycémie", "de": "Blutzucker"},
    "weight": {"en": "weight", "hi": "वजन", "hi-Latn": "wazan", "es": "peso", "fr": "poids", "de": "Gewicht"},
    "pain_score": {"en": "pain level", "hi": "दर्द का स्तर", "hi-Latn": "dard ka level", "es": "nivel de dolor", "fr": "niveau de douleur", "de": "Schmerzstärke"},
    # event kinds
    "pain": {"en": "pain", "hi": "दर्द", "hi-Latn": "dard", "es": "dolor", "fr": "douleur", "de": "Schmerzen"},
    "symptom": {"en": "symptom", "hi": "लक्षण", "hi-Latn": "lakshan", "es": "síntoma", "fr": "symptôme", "de": "Symptom"},
    "fall_reported": {"en": "reported fall", "hi": "बताया गया गिरना", "hi-Latn": "bataya gaya girna", "es": "caída informada", "fr": "chute signalée", "de": "gemeldeter Sturz"},
    "fall_detected": {"en": "detected fall", "hi": "पहचाना गया गिरना", "hi-Latn": "detect hua girna", "es": "caída detectada", "fr": "chute détectée", "de": "erkannter Sturz"},
    "measurement": {"en": "measurement", "hi": "माप", "hi-Latn": "measurement", "es": "medición", "fr": "mesure", "de": "Messung"},
    "medication_taken": {"en": "medication", "hi": "दवा", "hi-Latn": "dawai", "es": "medicación", "fr": "médicament", "de": "Medikament"},
    "emergency": {"en": "emergency", "hi": "आपातकाल", "hi-Latn": "emergency", "es": "emergencia", "fr": "urgence", "de": "Notfall"},
    "posture_alert": {"en": "posture reminder", "hi": "मुद्रा अनुस्मारक", "hi-Latn": "posture reminder", "es": "aviso de postura", "fr": "rappel de posture", "de": "Haltungshinweis"},
    "inactivity": {"en": "inactivity", "hi": "निष्क्रियता", "hi-Latn": "inactivity", "es": "inactividad", "fr": "inactivité", "de": "Inaktivität"},
    "check_in": {"en": "check-in", "hi": "हालचाल", "hi-Latn": "check-in", "es": "comprobación", "fr": "vérification", "de": "Nachfrage"},
    "escalation": {"en": "alert", "hi": "अलर्ट", "hi-Latn": "alert", "es": "alerta", "fr": "alerte", "de": "Alarm"},
    "allergy": {"en": "allergies", "hi": "एलर्जी", "hi-Latn": "allergy", "es": "alergias", "fr": "allergies", "de": "Allergien"},
    "note": {"en": "note", "hi": "नोट", "hi-Latn": "note", "es": "nota", "fr": "note", "de": "Notiz"},
    # concepts
    "c:pain": {"en": "pain", "hi": "दर्द", "hi-Latn": "dard", "es": "dolor", "fr": "douleur", "de": "Schmerzen"},
    "c:headache": {"en": "headache", "hi": "सिरदर्द", "hi-Latn": "sir dard", "es": "dolor de cabeza", "fr": "mal de tête", "de": "Kopfschmerzen"},
    "c:chest_pain": {"en": "chest pain", "hi": "सीने में दर्द", "hi-Latn": "seene mein dard", "es": "dolor en el pecho", "fr": "douleur thoracique", "de": "Brustschmerzen"},
    "c:breathing_difficulty": {"en": "difficulty breathing", "hi": "सांस लेने में तकलीफ", "hi-Latn": "saans lene mein takleef", "es": "dificultad para respirar", "fr": "difficulté à respirer", "de": "Atemnot"},
    "c:stroke_signs": {"en": "possible stroke signs", "hi": "संभावित स्ट्रोक के लक्षण", "hi-Latn": "stroke ke sambhavit lakshan", "es": "posibles signos de ictus", "fr": "signes possibles d'AVC", "de": "mögliche Schlaganfallzeichen"},
    "c:severe_bleeding": {"en": "severe bleeding", "hi": "बहुत खून बहना", "hi-Latn": "bahut khoon behna", "es": "sangrado abundante", "fr": "saignement abondant", "de": "starke Blutung"},
    "c:bleeding": {"en": "bleeding", "hi": "खून बहना", "hi-Latn": "khoon behna", "es": "sangrado", "fr": "saignement", "de": "Blutung"},
    "c:unconscious": {"en": "unresponsiveness", "hi": "बेहोशी", "hi-Latn": "behoshi", "es": "inconsciencia", "fr": "perte de connaissance", "de": "Bewusstlosigkeit"},
    "c:seizure": {"en": "seizure", "hi": "दौरा", "hi-Latn": "daura", "es": "convulsión", "fr": "convulsion", "de": "Krampfanfall"},
    "c:self_harm": {"en": "thoughts of self-harm", "hi": "खुद को नुकसान पहुंचाने के विचार", "hi-Latn": "khud ko nuksan pahunchane ke vichar", "es": "pensamientos de autolesión", "fr": "idées d'automutilation", "de": "Gedanken an Selbstverletzung"},
    "c:anaphylaxis_signs": {"en": "signs of a severe allergic reaction", "hi": "गंभीर एलर्जी प्रतिक्रिया के लक्षण", "hi-Latn": "gambhir allergy reaction ke lakshan", "es": "signos de reacción alérgica grave", "fr": "signes de réaction allergique grave", "de": "Zeichen einer schweren allergischen Reaktion"},
    "c:poisoning": {"en": "possible poisoning or overdose", "hi": "संभावित ज़हर या ओवरडोज़", "hi-Latn": "sambhavit zehar ya overdose", "es": "posible intoxicación o sobredosis", "fr": "intoxication ou surdose possible", "de": "mögliche Vergiftung oder Überdosis"},
    "c:head_injury": {"en": "head injury", "hi": "सिर पर चोट", "hi-Latn": "sir par chot", "es": "golpe en la cabeza", "fr": "choc à la tête", "de": "Kopfverletzung"},
    "c:fall": {"en": "fall", "hi": "गिरना", "hi-Latn": "girna", "es": "caída", "fr": "chute", "de": "Sturz"},
    "c:dizziness": {"en": "dizziness", "hi": "चक्कर", "hi-Latn": "chakkar", "es": "mareo", "fr": "vertiges", "de": "Schwindel"},
    "c:nausea": {"en": "nausea", "hi": "मतली", "hi-Latn": "matli", "es": "náuseas", "fr": "nausée", "de": "Übelkeit"},
    "c:vomiting": {"en": "vomiting", "hi": "उल्टी", "hi-Latn": "ulti", "es": "vómitos", "fr": "vomissements", "de": "Erbrechen"},
    "c:fever": {"en": "fever", "hi": "बुखार", "hi-Latn": "bukhar", "es": "fiebre", "fr": "fièvre", "de": "Fieber"},
    "c:cough": {"en": "cough", "hi": "खांसी", "hi-Latn": "khansi", "es": "tos", "fr": "toux", "de": "Husten"},
    "c:sore_throat": {"en": "sore throat", "hi": "गले में दर्द", "hi-Latn": "gale mein dard", "es": "dolor de garganta", "fr": "mal de gorge", "de": "Halsschmerzen"},
    "c:fatigue": {"en": "tiredness", "hi": "थकान", "hi-Latn": "thakan", "es": "cansancio", "fr": "fatigue", "de": "Müdigkeit"},
    "c:rash": {"en": "rash or itching", "hi": "दाने या खुजली", "hi-Latn": "daane ya khujli", "es": "sarpullido o picor", "fr": "éruption ou démangeaisons", "de": "Ausschlag oder Juckreiz"},
    "c:burn": {"en": "burn", "hi": "जलना", "hi-Latn": "jalna", "es": "quemadura", "fr": "brûlure", "de": "Verbrennung"},
    "c:cut": {"en": "cut", "hi": "कटना", "hi-Latn": "katna", "es": "corte", "fr": "coupure", "de": "Schnittwunde"},
    "c:sprain": {"en": "sprain", "hi": "मोच", "hi-Latn": "moch", "es": "esguince", "fr": "entorse", "de": "Verstauchung"},
    "c:swelling": {"en": "swelling", "hi": "सूजन", "hi-Latn": "sujan", "es": "hinchazón", "fr": "gonflement", "de": "Schwellung"},
    "c:numbness": {"en": "numbness or tingling", "hi": "सुन्नपन या झुनझुनी", "hi-Latn": "sunnpan ya jhunjhuni", "es": "entumecimiento u hormigueo", "fr": "engourdissement ou fourmillements", "de": "Taubheit oder Kribbeln"},
    "c:confusion": {"en": "confusion", "hi": "भ्रम", "hi-Latn": "confusion", "es": "confusión", "fr": "confusion", "de": "Verwirrtheit"},
    "c:anxiety": {"en": "anxiety", "hi": "घबराहट", "hi-Latn": "ghabrahat", "es": "ansiedad", "fr": "anxiété", "de": "Angst"},
    "c:insomnia": {"en": "trouble sleeping", "hi": "नींद न आना", "hi-Latn": "neend na aana", "es": "dificultad para dormir", "fr": "troubles du sommeil", "de": "Schlafprobleme"},
    "c:diarrhea": {"en": "diarrhoea", "hi": "दस्त", "hi-Latn": "dast", "es": "diarrea", "fr": "diarrhée", "de": "Durchfall"},
    "c:posture": {"en": "posture", "hi": "मुद्रा", "hi-Latn": "posture", "es": "postura", "fr": "posture", "de": "Haltung"},
}


def _placeholders(s: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(s) if f}


def say(key: str, lang: str, **params: Any) -> str:
    """Render a phrase. Missing language -> KeyError (caught by tests, never
    silently falls back to another language)."""
    tmpl = P[key][lang]
    return tmpl.format(**params) if params or _placeholders(tmpl) else tmpl


def name(key: str, lang: str) -> str:
    entry = NAMES.get(key)
    if not entry:
        return key.replace("c:", "").replace("_", " ")
    return entry[lang]


def validate() -> list[str]:
    """Return a list of problems (used by tests)."""
    problems: list[str] = []
    for key, by_lang in P.items():
        ref = _placeholders(by_lang.get("en", ""))
        for lang in SUPPORTED_LANGS:
            if lang not in by_lang:
                problems.append(f"phrase {key} missing {lang}")
            elif _placeholders(by_lang[lang]) != ref:
                problems.append(f"phrase {key} placeholder mismatch in {lang}")
    for key, by_lang in NAMES.items():
        for lang in SUPPORTED_LANGS:
            if lang not in by_lang:
                problems.append(f"name {key} missing {lang}")
    return problems
