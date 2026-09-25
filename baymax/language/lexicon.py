"""Multilingual lexicons for the rule-based NLU model (nlu_lexicon_v1).

Format: for each key, a dict lang -> "|"-separated phrases. Phrases are written
in *normalised* form (see baymax.language.normalize): lower case, Latin
accents removed, ß->ss, hyphens as spaces, Devanagari nukta removed and
chandrabindu folded to anusvara. A trailing "*" on a single-word phrase means
prefix match (German compounds: "kopfschmerz*").

Coverage of every Concept in every Lang is enforced by tests.
"""
from __future__ import annotations

from ..ontology import BodyRegion, Concept

L = dict[str, str]

# --------------------------------------------------------------------------
# Concepts
# --------------------------------------------------------------------------
CONCEPTS: dict[Concept, L] = {
    Concept.PAIN: {
        "en": "pain|painful|hurts|hurt|hurting|ache|aches|aching|sore|it hurts|in pain",
        "hi-Latn": "dard|dukh raha|dukh rahi|pida|peeda|dard ho raha|dard hai|chot lagi|chot",
        "hi": "दर्द|पीडा|दुख रहा|दुख रही|चोट लगी|चोट",
        "es": "dolor|duele|me duele|adolorido|adolorida|dolorido",
        "fr": "douleur|douleurs|j'ai mal|mal a|mal au|mal aux|fait mal|me fait mal",
        "de": "schmerz*|tut weh|weh|schmerzt",
    },
    Concept.HEADACHE: {
        "en": "headache|migraine",
        "hi-Latn": "sir dard|sar dard|sirdard|sardard",
        "hi": "सिर दर्द|सिरदर्द|सर दर्द",
        "es": "dolor de cabeza|jaqueca|migrana",
        "fr": "mal de tete|mal a la tete|migraine",
        "de": "kopfschmerz*|kopfweh|migrane",
    },
    Concept.CHEST_PAIN: {
        "en": "chest pain|pain in my chest|chest hurts|chest is hurting|tightness in my chest|chest tightness|chest feels tight|pressure in my chest|chest pressure|crushing chest|heart attack|having a heart attack",
        "hi-Latn": "seene me dard|seene mein dard|chhati me dard|chhati mein dard|chest pain|dil ka daura|heart attack|seene mein dabav|seene me dabav",
        "hi": "सीने में दर्द|छाती में दर्द|सीने में दबाव|दिल का दौरा|हार्ट अटैक",
        "es": "dolor de pecho|dolor en el pecho|me duele el pecho|opresion en el pecho|ataque al corazon|infarto",
        "fr": "douleur thoracique|douleur a la poitrine|mal a la poitrine|mal dans la poitrine|oppression thoracique|crise cardiaque|infarctus",
        "de": "brustschmerz*|schmerzen in der brust|druck auf der brust|engegefuhl in der brust|herzinfarkt",
    },
    Concept.BREATHING_DIFFICULTY: {
        "en": "can't breathe|cant breathe|cannot breathe|can not breathe|difficulty breathing|trouble breathing|hard to breathe|short of breath|shortness of breath|struggling to breathe|not breathing|isn't breathing|isnt breathing|stopped breathing|choking|gasping for air|can't catch my breath",
        "hi-Latn": "saans nahi aa rahi|saans nahi aa raha|saans nahi le pa raha|saans nahi le pa rahi|saans lene me takleef|saans lene mein takleef|saans lene mein dikkat|saans lene me dikkat|saans phool rahi|saans nahi|sans nahi|dam ghut raha|dum ghut raha",
        "hi": "सांस नहीं आ रही|सांस नहीं आ रहा|सांस लेने में तकलीफ|सांस लेने में दिक्कत|सांस फूल रही|सांस नहीं|दम घुट रहा",
        "es": "no puedo respirar|dificultad para respirar|me falta el aire|falta de aire|me ahogo|no respira|se esta ahogando",
        "fr": "je ne peux pas respirer|j'arrive pas a respirer|je n'arrive pas a respirer|difficulte a respirer|du mal a respirer|ne respire plus|ne respire pas|je m'etouffe|il s'etouffe|elle s'etouffe",
        "de": "kann nicht atmen|atemnot|keine luft|bekomme keine luft|kriege keine luft|atmet nicht|ersticke|erstickt",
    },
    Concept.STROKE_SIGNS: {
        "en": "face drooping|face is drooping|drooping face|face droop|slurred speech|slurring|can't move my arm|cant move my arm|can't feel my arm|one side of my body|arm is weak|sudden weakness|can't speak|cant speak|having a stroke|a stroke",
        "hi-Latn": "muh tedha|munh tedha|chehra latak|zubaan ladkhada|zuban ladkhada|bolne me dikkat|bolne mein dikkat|haath nahi hil raha|lakwa|stroke",
        "hi": "मुंह टेढा|चेहरा लटक|जुबान लडखडा|बोलने में दिक्कत|हाथ नहीं हिल रहा|लकवा|स्ट्रोक",
        "es": "cara caida|habla arrastrada|no puedo mover el brazo|derrame cerebral|ictus",
        "fr": "visage affaisse|bouche de travers|difficulte a parler|je ne peux plus bouger le bras|avc",
        "de": "gesicht hangt|hangender mundwinkel|verwaschene sprache|kann den arm nicht bewegen|schlaganfall",
    },
    Concept.SEVERE_BLEEDING: {
        "en": "bleeding a lot|bleeding heavily|heavy bleeding|won't stop bleeding|wont stop bleeding|bleeding won't stop|bleeding wont stop|lots of blood|so much blood|blood everywhere|gushing blood|severe bleeding",
        "hi-Latn": "bahut khoon|khoon ruk nahi raha|khoon band nahi ho raha|bahut khoon beh raha",
        "hi": "बहुत खून|खून रुक नहीं रहा|खून बंद नहीं हो रहा",
        "es": "sangra mucho|mucha sangre|no para de sangrar|hemorragia",
        "fr": "saigne beaucoup|beaucoup de sang|saignement abondant|hemorragie|n'arrete pas de saigner",
        "de": "blutet stark|viel blut|starke blutung|hort nicht auf zu bluten",
    },
    Concept.BLEEDING: {
        "en": "bleeding|i'm bleeding|bleeds",
        "hi-Latn": "khoon|khoon nikal|khoon aa raha",
        "hi": "खून|खून निकल",
        "es": "sangre|sangrando|sangro|sangra",
        "fr": "saigne|saignement|sang",
        "de": "blute|blutet|blutung|blutend",
    },
    Concept.UNCONSCIOUS: {
        "en": "unconscious|passed out|not responding|unresponsive|won't wake up|wont wake up|fainted|blacked out|collapsed",
        "hi-Latn": "behosh|hosh nahi|hosh me nahi|uth nahi raha|uth nahi rahi|jawab nahi de raha|jawab nahi de rahi",
        "hi": "बेहोश|होश नहीं|उठ नहीं रहा|उठ नहीं रही|जवाब नहीं दे रहा|जवाब नहीं दे रही",
        "es": "inconsciente|se desmayo|me desmaye|no responde|no despierta|se desplomo",
        "fr": "inconscient|inconsciente|evanoui|evanouie|s'est evanoui|ne repond pas|ne se reveille pas|s'est effondre",
        "de": "bewusstlos|ohnmachtig|reagiert nicht|wacht nicht auf|zusammengebrochen",
    },
    Concept.SEIZURE: {
        "en": "seizure|having a fit|convulsing|convulsions|fitting",
        "hi-Latn": "daura pad|daura aa|mirgi|jhatke aa rahe",
        "hi": "दौरा पड|दौरा आ|मिर्गी|झटके आ रहे",
        "es": "convulsion|convulsiones|convulsionando|ataque epileptico",
        "fr": "convulsion*|crise d'epilepsie|crise convulsive",
        "de": "krampfanfall|epileptischer anfall|krampft",
    },
    Concept.SELF_HARM: {
        "en": "kill myself|end my life|want to die|suicide|suicidal|hurt myself|harm myself|don't want to live|dont want to live|no reason to live|better off dead|cut myself on purpose",
        "hi-Latn": "mar jana chahta|mar jana chahti|jeena nahi chahta|jeena nahi chahti|khudkushi|aatmahatya|khud ko nuksan",
        "hi": "मर जाना चाहता|मर जाना चाहती|जीना नहीं चाहता|जीना नहीं चाहती|आत्महत्या|खुदकुशी|खुद को नुकसान",
        "es": "suicidarme|quiero morir|matarme|quitarme la vida|hacerme dano|suicidio",
        "fr": "me suicider|suicide|envie de mourir|veux mourir|me tuer|me faire du mal|en finir",
        "de": "mich umbringen|selbstmord|suizid|will sterben|nicht mehr leben|mir etwas antun",
    },
    Concept.ANAPHYLAXIS_SIGNS: {
        "en": "throat is swelling|throat swelling|throat closing|throat is closing|tongue swelling|tongue is swelling|lips swelling|anaphylaxis|anaphylactic",
        "hi-Latn": "gala sooj|gale me sujan|gale mein sujan|jeebh sooj|hont sooj",
        "hi": "गला सूज|गले में सूजन|जीभ सूज|होंठ सूज",
        "es": "se me cierra la garganta|garganta hinchada|lengua hinchada|anafilaxia|reaccion alergica grave",
        "fr": "gorge qui gonfle|gorge gonflee|langue gonflee|anaphylaxie|choc anaphylactique",
        "de": "hals schwillt|zunge schwillt|anaphylaxie|anaphylaktisch|allergischer schock",
    },
    Concept.POISONING: {
        "en": "overdose|overdosed|poisoned|swallowed bleach|drank bleach|took too many pills|too many pills|poisoning",
        "hi-Latn": "zeher|zehar|jahar|bahut saari goliyan|zyada goliyan|bahut goliyan kha li",
        "hi": "जहर|बहुत सारी गोलियां|ज्यादा गोलियां",
        "es": "sobredosis|envenenado|envenenada|envenenamiento|demasiadas pastillas|tome lejia|bebio lejia",
        "fr": "surdose|overdose|empoisonne|empoisonnee|intoxication|trop de comprimes|trop de medicaments|bu de l'eau de javel",
        "de": "uberdosis|vergiftet|vergiftung|zu viele tabletten",
    },
    Concept.HEAD_INJURY: {
        "en": "hit my head|banged my head|bumped my head|head injury|hit his head|hit her head|hit their head|hit the head",
        "hi-Latn": "sir pe chot|sar pe chot|sir mein chot|sir me chot|sar me chot|sir par chot",
        "hi": "सिर पर चोट|सिर में चोट|सर पर चोट",
        "es": "golpe en la cabeza|me golpee la cabeza|me di en la cabeza|se golpeo la cabeza",
        "fr": "cogne la tete|coup a la tete|choc a la tete|tape la tete",
        "de": "kopf gestossen|kopf angeschlagen|kopfverletzung|auf den kopf gefallen",
    },
    Concept.FALL: {
        "en": "i fell|fell down|i've fallen|i have fallen|fallen down|took a fall|slipped|tripped|fell over|fell off|he fell|she fell|they fell|has fallen|can't get up|cant get up|cannot get up|i fall",
        "hi-Latn": "gir gaya|gir gayi|gir gai|gir pada|gir padi|fisal gaya|fisal gayi|uth nahi pa raha|uth nahi pa rahi|utha nahi ja raha|gir gya",
        "hi": "गिर गया|गिर गई|गिर गयी|गिर पडा|गिर पडी|फिसल गया|फिसल गई|उठ नहीं पा रहा|उठ नहीं पा रही",
        "es": "me cai|me he caido|se cayo|se ha caido|me resbale|tropece|no puedo levantarme",
        "fr": "je suis tombe|je suis tombee|suis tombe|est tombe|est tombee|j'ai glisse|j'ai trebuche|je ne peux pas me relever|je n'arrive pas a me relever",
        "de": "bin gefallen|hingefallen|gesturzt|ausgerutscht|gestolpert|komme nicht hoch|kann nicht aufstehen|ist gefallen|ist gesturzt",
    },
    Concept.DIZZINESS: {
        "en": "dizzy|dizziness|lightheaded|light headed|room is spinning|vertigo|feel faint",
        "hi-Latn": "chakkar|sar ghoom|sir ghoom|sir ghum",
        "hi": "चक्कर|सिर घूम",
        "es": "mareo|mareado|mareada|vertigo",
        "fr": "vertige*|etourdi*|tete qui tourne",
        "de": "schwindel*|schwindlig|benommen",
    },
    Concept.NAUSEA: {
        "en": "nausea|nauseous|feel sick|feeling sick|queasy",
        "hi-Latn": "ji machla|jee machla|ji michla|ulti jaisa|matli",
        "hi": "जी मिचला|उल्टी जैसा|मतली",
        "es": "nauseas|ganas de vomitar|revuelto el estomago",
        "fr": "nausee*|envie de vomir|mal au coeur",
        "de": "ubel|ubelkeit|mir ist schlecht",
    },
    Concept.VOMITING: {
        "en": "vomit|vomiting|vomited|threw up|throwing up|puking|puked",
        "hi-Latn": "ulti ho|ulti hui|ulti kar|ultiyan",
        "hi": "उल्टी हो|उल्टी हुई|उल्टियां",
        "es": "vomito|vomitando|vomitado|vomite",
        "fr": "vomi|vomis|vomit|vomissement*",
        "de": "erbrech*|erbrochen|ubergeben|kotzen",
    },
    Concept.FEVER: {
        "en": "fever|feverish|high temperature|temperature is high",
        "hi-Latn": "bukhar|bukhaar|taap",
        "hi": "बुखार|ताप",
        "es": "fiebre|calentura",
        "fr": "fievre|fievreux",
        "de": "fieber",
    },
    Concept.COUGH: {
        "en": "cough|coughing|coughed",
        "hi-Latn": "khansi|khaansi|khasi",
        "hi": "खांसी",
        "es": "tos|tosiendo|toso",
        "fr": "toux|tousse|tousser",
        "de": "husten|huste",
    },
    Concept.SORE_THROAT: {
        "en": "sore throat|throat hurts|throat is sore",
        "hi-Latn": "gale me dard|gale mein dard|gala kharab|gala dukh",
        "hi": "गले में दर्द|गला खराब",
        "es": "dolor de garganta|me duele la garganta",
        "fr": "mal a la gorge|mal de gorge",
        "de": "halsschmerz*|halsweh",
    },
    Concept.FATIGUE: {
        "en": "tired|exhausted|fatigue|fatigued|no energy|worn out",
        "hi-Latn": "thakan|thaka hua|thaki hui|kamzori|thak gaya|thak gayi",
        "hi": "थकान|थका हुआ|थकी हुई|कमजोरी|थक गया|थक गई",
        "es": "cansado|cansada|agotado|agotada|fatiga",
        "fr": "fatigue|fatiguee|epuise|epuisee|creve",
        "de": "mude|erschopft|mudigkeit",
    },
    Concept.RASH: {
        "en": "rash|hives|itchy|itching",
        "hi-Latn": "daane|khujli|chakatte",
        "hi": "दाने|खुजली|चकत्ते",
        "es": "sarpullido|erupcion|ronchas|picazon|comezon",
        "fr": "eruption|boutons|demangeaison*|urticaire",
        "de": "ausschlag|juckreiz|juckt|quaddeln",
    },
    Concept.BURN: {
        "en": "burn|burned|burnt|scalded|burned myself",
        "hi-Latn": "jal gaya|jal gayi|jal gaya hai|haath jal",
        "hi": "जल गया|जल गई|जला|जलने|जल जाने",
        "es": "quemadura|me queme|quemado|quemada",
        "fr": "brulure|je me suis brule|brule|brulee",
        "de": "verbrannt|verbrennung|verbruht",
    },
    Concept.CUT: {
        "en": "cut myself|cut my|a cut|deep cut",
        "hi-Latn": "kat gaya|kat gayi|kat gaya hai",
        "hi": "कट गया|कट गई",
        "es": "corte|me corte|cortada",
        "fr": "coupure|je me suis coupe|je me suis coupee",
        "de": "schnittwunde|geschnitten|schnitt",
    },
    Concept.SPRAIN: {
        "en": "sprain|sprained|twisted my|rolled my ankle",
        "hi-Latn": "moch|moch aa",
        "hi": "मोच",
        "es": "esguince|torcedura|me torci",
        "fr": "entorse|foulure|tordu",
        "de": "verstaucht|verstauchung|umgeknickt",
    },
    Concept.SWELLING: {
        "en": "swollen|swelling|swelled",
        "hi-Latn": "sujan|sooj gaya|sooj gayi|soojan",
        "hi": "सूजन|सूज गया|सूज गई",
        "es": "hinchado|hinchada|hinchazon",
        "fr": "gonfle*|enfle*",
        "de": "geschwollen|schwellung",
    },
    Concept.NUMBNESS: {
        "en": "numb|numbness|tingling|pins and needles",
        "hi-Latn": "sunn|jhunjhuni",
        "hi": "सुन्न|झुनझुनी",
        "es": "entumecido|entumecida|adormecido|hormigueo",
        "fr": "engourdi*|fourmillement*",
        "de": "taub|kribbeln",
    },
    Concept.CONFUSION: {
        "en": "confused|disoriented|don't know where i am",
        "hi-Latn": "samajh nahi aa raha|confuse|pata nahi main kahan",
        "hi": "समझ नहीं आ रहा|उलझन",
        "es": "confundido|confundida|desorientado|desorientada",
        "fr": "confus|confuse|desoriente|desorientee",
        "de": "verwirrt|desorientiert",
    },
    Concept.ANXIETY: {
        "en": "anxious|anxiety|panic|panic attack|scared|stressed|heart racing",
        "hi-Latn": "ghabrahat|ghabra|dar lag raha|tension|chinta",
        "hi": "घबराहट|घबरा|डर लग रहा|चिंता",
        "es": "ansiedad|ansioso|ansiosa|ataque de panico|nervioso|nerviosa",
        "fr": "angoisse|anxieux|anxieuse|crise de panique|stresse|stressee",
        "de": "angst|panikattacke|gestresst|nervos",
    },
    Concept.INSOMNIA: {
        "en": "can't sleep|cant sleep|insomnia|couldn't sleep|couldnt sleep|trouble sleeping",
        "hi-Latn": "neend nahi|so nahi pa|so nahi paya|so nahi payi",
        "hi": "नींद नहीं|सो नहीं पा",
        "es": "insomnio|no puedo dormir|no pude dormir",
        "fr": "insomnie|je n'arrive pas a dormir|pas dormi",
        "de": "schlaflos*|kann nicht schlafen|schlafe schlecht",
    },
    Concept.DIARRHEA: {
        "en": "diarrhea|diarrhoea|loose motion|loose motions",
        "hi-Latn": "dast|loose motion|loose motions|pet kharab",
        "hi": "दस्त|पेट खराब",
        "es": "diarrea",
        "fr": "diarrhee",
        "de": "durchfall",
    },
    Concept.POSTURE: {
        "en": "posture|slouching|slouch",
        "hi-Latn": "posture|jhuk kar baithna",
        "hi": "मुद्रा|झुककर बैठना",
        "es": "postura|encorvado",
        "fr": "posture|voute",
        "de": "haltung|krumm sitzen",
    },
}

# Derived concepts: generic pain in a region implies a specific concept.
PAIN_REGION_DERIVED: dict[BodyRegion, Concept] = {
    BodyRegion.CHEST: Concept.CHEST_PAIN,
    BodyRegion.HEAD: Concept.HEADACHE,
    BodyRegion.THROAT: Concept.SORE_THROAT,
}

# Concepts matched across ALL languages regardless of detected language
# (recall matters more than precision for these).
CROSS_LANGUAGE_CONCEPTS = {
    Concept.CHEST_PAIN, Concept.BREATHING_DIFFICULTY, Concept.STROKE_SIGNS,
    Concept.SEVERE_BLEEDING, Concept.UNCONSCIOUS, Concept.SEIZURE,
    Concept.SELF_HARM, Concept.ANAPHYLAXIS_SIGNS, Concept.POISONING,
}

# --------------------------------------------------------------------------
# Body regions
# --------------------------------------------------------------------------
BODY: dict[BodyRegion, L] = {
    BodyRegion.HEAD: {"en": "head", "hi-Latn": "sir|sar", "hi": "सिर|सर", "es": "cabeza", "fr": "tete", "de": "kopf"},
    BodyRegion.FACE: {"en": "face", "hi-Latn": "chehra|chehre", "hi": "चेहरा|चेहरे", "es": "cara", "fr": "visage", "de": "gesicht"},
    BodyRegion.EYE: {"en": "eye|eyes", "hi-Latn": "aankh|aankhon|ankh", "hi": "आंख|आंखों", "es": "ojo|ojos", "fr": "oeil|yeux", "de": "auge|augen"},
    BodyRegion.EAR: {"en": "ear|ears", "hi-Latn": "kaan", "hi": "कान", "es": "oido|oreja", "fr": "oreille|oreilles", "de": "ohr|ohren"},
    BodyRegion.MOUTH: {"en": "mouth|tooth|teeth|jaw", "hi-Latn": "munh|daant|jabda", "hi": "मुंह|दांत|जबडा", "es": "boca|diente|muela|mandibula", "fr": "bouche|dent|dents|machoire", "de": "mund|zahn|zahne|kiefer"},
    BodyRegion.THROAT: {"en": "throat", "hi-Latn": "gala|gale", "hi": "गला|गले", "es": "garganta", "fr": "gorge", "de": "hals"},
    BodyRegion.NECK: {"en": "neck", "hi-Latn": "gardan", "hi": "गर्दन", "es": "cuello", "fr": "cou|nuque", "de": "nacken"},
    BodyRegion.SHOULDER: {"en": "shoulder|shoulders", "hi-Latn": "kandha|kandhe", "hi": "कंधा|कंधे", "es": "hombro|hombros", "fr": "epaule|epaules", "de": "schulter|schultern"},
    BodyRegion.CHEST: {"en": "chest", "hi-Latn": "seena|seene|chhati|chest", "hi": "सीना|सीने|छाती", "es": "pecho", "fr": "poitrine|thorax", "de": "brust"},
    BodyRegion.ABDOMEN: {"en": "stomach|belly|abdomen|tummy", "hi-Latn": "pet", "hi": "पेट", "es": "estomago|barriga|vientre|abdomen|panza", "fr": "ventre|estomac", "de": "bauch|magen"},
    BodyRegion.BACK: {"en": "back|lower back|spine", "hi-Latn": "kamar|peeth", "hi": "कमर|पीठ", "es": "espalda", "fr": "dos", "de": "rucken"},
    BodyRegion.ARM: {"en": "arm|arms", "hi-Latn": "baah|baazu|bazu", "hi": "बांह|बाजू", "es": "brazo|brazos", "fr": "bras", "de": "arm|arme"},
    BodyRegion.ELBOW: {"en": "elbow", "hi-Latn": "kohni", "hi": "कोहनी", "es": "codo", "fr": "coude", "de": "ellbogen"},
    BodyRegion.WRIST: {"en": "wrist", "hi-Latn": "kalai", "hi": "कलाई", "es": "muneca", "fr": "poignet", "de": "handgelenk"},
    BodyRegion.HAND: {"en": "hand|hands|finger|fingers|thumb", "hi-Latn": "haath|hath|ungli|ungliyan", "hi": "हाथ|उंगली", "es": "mano|manos|dedo|dedos", "fr": "main|mains|doigt|doigts", "de": "hand|hande|finger|daumen"},
    BodyRegion.HIP: {"en": "hip|hips", "hi-Latn": "kulha|kulhe", "hi": "कूल्हा|कूल्हे", "es": "cadera", "fr": "hanche", "de": "hufte"},
    BodyRegion.LEG: {"en": "leg|legs|thigh", "hi-Latn": "pair|taang|tang|jangh", "hi": "पैर|टांग|जांघ", "es": "pierna|piernas|muslo", "fr": "jambe|jambes|cuisse", "de": "bein|beine|oberschenkel"},
    BodyRegion.KNEE: {"en": "knee|knees", "hi-Latn": "ghutna|ghutne", "hi": "घुटना|घुटने", "es": "rodilla|rodillas", "fr": "genou|genoux", "de": "knie"},
    BodyRegion.ANKLE: {"en": "ankle|ankles", "hi-Latn": "takhna|takhne|edi", "hi": "टखना|टखने|एडी", "es": "tobillo|tobillos", "fr": "cheville|chevilles", "de": "knochel|sprunggelenk"},
    BodyRegion.FOOT: {"en": "foot|feet|toe|toes", "hi-Latn": "paon|panja|pao", "hi": "पांव|पंजा", "es": "pie|pies", "fr": "pied|pieds|orteil", "de": "fuss|fusse|zeh|zehen"},
}

# --------------------------------------------------------------------------
# Intent cue phrases
# --------------------------------------------------------------------------
EMERGENCY_STRONG: L = {  # anywhere in a clause, matched across all languages
    "en": "call an ambulance|call ambulance|call 911|call 999|call 112|call 108|call emergency|emergency services|somebody help|someone help|help me please|please help me|please help|call for help|i need an ambulance|get an ambulance|send an ambulance|this is an emergency|it's an emergency|its an emergency|medical emergency|sos",
    "hi-Latn": "bachao|madad karo|ambulance bulao|ambulance bula do|emergency hai|koi bachao|help karo|jaldi madad",
    "hi": "बचाओ|मदद करो|एम्बुलेंस बुलाओ|एंबुलेंस बुलाओ|आपातकाल|इमरजेंसी है|जल्दी मदद",
    "es": "socorro|auxilio|llama a una ambulancia|llamen a una ambulancia|llama al 112|llama al 911|es una emergencia|emergencia medica|ayudenme|ayudame por favor",
    "fr": "au secours|a l'aide|aidez moi|appelez une ambulance|appelle une ambulance|appelez le samu|appelle le samu|c'est une urgence|urgence medicale|appelez le 15|appelle le 15",
    "de": "hilfe hilfe|hilf mir|helft mir|ruf einen krankenwagen|ruft einen krankenwagen|krankenwagen|notarzt|rettungswagen|es ist ein notfall|notfall",
}
EMERGENCY_EXACT: L = {  # whole utterance (after stripping fillers)
    "en": "help|help help|help me|emergency|ambulance",
    "hi-Latn": "madad|help|bachao",
    "hi": "मदद|बचाओ",
    "es": "ayuda|ayudame|emergencia",
    "fr": "aide|aidez moi|urgence",
    "de": "hilfe",
}
# "I need help" type phrases: emergency only if nothing follows (e.g. not "... with my homework")
HELP_OPEN: L = {
    "en": "i need help|need help|i need some help",
    "hi-Latn": "madad chahiye|help chahiye|mujhe madad chahiye",
    "hi": "मदद चाहिए|मुझे मदद चाहिए",
    "es": "necesito ayuda",
    "fr": "j'ai besoin d'aide|besoin d'aide",
    "de": "ich brauche hilfe|brauche hilfe",
}
HELP_CONTINUATION: L = {
    "en": "with|for|on|to|finding|setting|writing",
    "hi-Latn": "me|mein|ke|ki|liye",
    "hi": "में|के|की|लिए",
    "es": "con|para|en",
    "fr": "pour|avec|sur|a",
    "de": "bei|mit|fur|zu",
}
POSSIBLE_EMERGENCY: L = {
    "en": "i'm dying|im dying|i am dying|something is wrong|something's wrong|somethings wrong|i feel really bad|i'm hurt|im hurt|i am hurt|i got hurt|i'm not okay|im not okay|i am not okay|not ok|worst pain of my life|worst pain ever|is this an emergency",
    "hi-Latn": "main mar raha|main mar rahi|mar raha hoon|mar rahi hoon|kuch gadbad|tabiyat kharab|tabiyat bahut kharab|theek nahi lag raha|chot lag gayi",
    "hi": "मैं मर रहा|मैं मर रही|कुछ गडबड|तबीयत खराब|ठीक नहीं लग रहा|चोट लग गई",
    "es": "me estoy muriendo|algo anda mal|algo va mal|no me siento bien|estoy herido|estoy herida|me siento muy mal",
    "fr": "je meurs|je suis en train de mourir|quelque chose ne va pas|je ne me sens pas bien|je suis blesse|je suis blessee|ca ne va pas du tout",
    "de": "ich sterbe|etwas stimmt nicht|mir geht es nicht gut|mir gehts nicht gut|ich bin verletzt|mir geht es schlecht",
}
IDIOMS: L = {  # suppress emergency/pain readings inside these spans
    "en": "dying of laughter|dying laughing|dying to|killing it|killing time|gave me a heart attack|almost gave me a heart attack|nearly gave me a heart attack|heart attack on a plate|pain in the neck|help you|help him|help her|help them|help my|heartburn|i'm dead tired|dead tired|dying for",
    "hi-Latn": "hass hass ke mar|haste haste mar|hans hans ke mar",
    "hi": "हंस हंस के मर",
    "es": "muriendo de risa|morir de risa|muero de risa|me muero por",
    "fr": "mourir de rire|mort de rire|morte de rire|meurs d'envie",
    "de": "totlachen|tot gelacht|todmude",
}
FAKE_MARKERS: L = {
    "en": "prank|as a joke|just kidding|jk|fake|as a test|pretend|for fun|lol",
    "hi-Latn": "mazaak|mazak|prank|jhooth mooth|majak",
    "hi": "मजाक|झूठमूठ",
    "es": "broma|es broma|de mentira|falso",
    "fr": "blague|pour rire|faux|farce",
    "de": "scherz|spass|nur ein witz|witz|fake",
}
PAIN_EXCLAMATIONS: L = {
    "en": "ouch|ow|oww|ouchie|argh|aargh|ahh|aah|aahh|ugh|oof",
    "hi-Latn": "aah|aahh|ahh|uff|oof|ohh|oww|haye|ui|ui maa|aaah|ahhh",
    "hi": "आह|उफ|हाय|ओह|आउच|उई|उई मां",
    "es": "ay|ayy|auch|ayayay|ay ay|uy",
    "fr": "aie|ouille|ouch|ouf|aie aie",
    "de": "aua|autsch|au|aah|auweh",
}
USER_OK: L = {
    "en": "i'm ok|i'm okay|im ok|im okay|i am ok|i am okay|i'm fine|im fine|i am fine|i'm alright|i'm all right|im alright|i am alright|i'm good|im good|all good|no problem|i'm safe|false alarm|don't worry|dont worry|i'm not hurt|im not hurt|not hurt|all fine|everything is fine|i'm totally fine",
    "hi-Latn": "main theek hoon|main thik hoon|main thik hu|main theek hu|theek hoon|thik hoon|thik hu|theek hu|sab theek|sab thik|koi baat nahi|chinta mat karo|thik hai|theek hai|main thik|main theek",
    "hi": "मैं ठीक हूं|ठीक हूं|सब ठीक|कोई बात नहीं|चिंता मत करो|ठीक है|मैं ठीक",
    "es": "estoy bien|todo bien|no pasa nada|falsa alarma|no te preocupes|estoy perfectamente",
    "fr": "je vais bien|ca va|tout va bien|je suis ok|pas de souci|fausse alerte|ne t'inquiete pas|ca va bien",
    "de": "mir geht es gut|mir gehts gut|mir geht's gut|alles gut|alles okay|alles in ordnung|ich bin okay|fehlalarm|keine sorge|mir fehlt nichts",
}
USER_OK_EXACT: L = {
    "en": "ok|okay|fine|good|alright|i'm good",
    "hi-Latn": "theek|thik|thik hai|theek hai|sab badhiya",
    "hi": "ठीक|ठीक है",
    "es": "bien|vale|ok",
    "fr": "ok|ca va|bien",
    "de": "gut|okay|ok|passt",
}
CONFIRM: L = {
    "en": "yes|yeah|yep|yes please|do it|call them|call now|please call|notify them|send help|go ahead|yes call|call someone",
    "hi-Latn": "haan|han|haa|ji haan|haan karo|call karo|bulao|haan bulao",
    "hi": "हां|जी हां|कॉल करो|बुलाओ",
    "es": "si|si por favor|llama|llamalos|hazlo|avisa",
    "fr": "oui|oui s'il te plait|appelle|appelle les|vas y|previens les",
    "de": "ja|ja bitte|ruf an|mach das|bitte anrufen|benachrichtige sie",
}
CANCEL: L = {
    "en": "cancel|stop|don't call|dont call|do not call|never mind|nevermind|no|nope|abort|stop the alarm|don't alert|dont alert",
    "hi-Latn": "nahi|nahin|mat karo|call mat karo|ruko|cancel karo|rehne do|band karo|cancel",
    "hi": "नहीं|मत करो|रुको|रहने दो|कॉल मत करो|कैंसल|बंद करो",
    "es": "no|cancela|cancelar|no llames|para|detente|dejalo",
    "fr": "non|annule|annuler|n'appelle pas|arrete|stop|laisse tomber",
    "de": "nein|abbrechen|stopp|ruf nicht an|nicht anrufen|halt|lass es",
}
GREETING: L = {
    "en": "hello|hi|hey|good morning|good evening|good afternoon|hi baymax|hello baymax",
    "hi-Latn": "namaste|namaskar|kaise ho|hello",
    "hi": "नमस्ते|नमस्कार|कैसे हो",
    "es": "hola|buenos dias|buenas tardes|buenas noches",
    "fr": "bonjour|salut|bonsoir|coucou",
    "de": "hallo|guten morgen|guten tag|guten abend|servus|moin",
}
GOODBYE: L = {
    "en": "bye|goodbye|good night|that's all|thats all|thank you that's all|thanks that's all",
    "hi-Latn": "alvida|bas itna hi|shubh ratri|phir milte hain|bye",
    "hi": "अलविदा|बस इतना ही|शुभ रात्रि|फिर मिलते हैं",
    "es": "adios|hasta luego|buenas noches|eso es todo",
    "fr": "au revoir|bonne nuit|a plus tard|c'est tout",
    "de": "tschuss|auf wiedersehen|gute nacht|das war's|das wars",
}
QUESTION: L = {
    "en": "what|how|should i|can i|is it|why|when should|do i need|what should|what do i|what can",
    "hi-Latn": "kya|kaise|kyu|kyon|chahiye|kab|kya karu|kya karun|kya kare",
    "hi": "क्या|कैसे|क्यों|चाहिए|कब",
    "es": "que|como|debo|puedo|por que|cuando|que hago|que hacer",
    "fr": "que|quoi|comment|dois je|est ce|pourquoi|quand|que faire",
    "de": "was|wie|soll ich|warum|wann|kann ich|muss ich|was tun",
}
DIAGNOSIS_REQUEST: L = {
    "en": "diagnose|diagnosis|what do i have|what disease|what's wrong with me|whats wrong with me|do i have|is it cancer|am i sick with",
    "hi-Latn": "mujhe kya bimari|kaunsi bimari|kya bimari hai|diagnose|mujhe kya hua",
    "hi": "मुझे क्या बीमारी|कौन सी बीमारी|क्या बीमारी है|मुझे क्या हुआ",
    "es": "diagnostico|diagnosticame|que enfermedad|que tengo|tengo cancer",
    "fr": "diagnostic|diagnostique|quelle maladie|qu'est ce que j'ai|ai je un cancer",
    "de": "diagnose|diagnostiziere|welche krankheit|was habe ich|habe ich krebs",
}
DOSAGE_REQUEST: L = {
    "en": "how much should i take|how many pills|what dose|dosage|how many mg|how much ibuprofen|how much paracetamol|how much tylenol|how many tablets|double my dose|double the dose",
    "hi-Latn": "kitni goli|kitni dawai|kitna dose|dose kitna|kitni tablet",
    "hi": "कितनी गोली|कितनी दवा|कितना डोज|खुराक",
    "es": "que dosis|cuantas pastillas|cuanto debo tomar|cuantos mg|dosis",
    "fr": "quelle dose|combien de comprimes|combien dois je prendre|posologie|dose",
    "de": "welche dosis|wie viele tabletten|wie viel soll ich nehmen|dosierung|dosis",
}
RECALL: L = {
    "en": "what am i allergic to|my allergies|what do you remember|what did i tell you|what do you know about me|what was my|what were my|when did i last|when was my last|show my|my last|history of|how many times did i|did i take|have i taken|what is my latest|my latest|last reading",
    "hi-Latn": "mujhe kis se allergy|mujhe kis cheez se allergy|meri allergy|kya yaad hai|kitna tha|kab li thi|pichli baar|mera last|meri last|kitni baar|history dikhao",
    "hi": "मुझे किससे एलर्जी|मेरी एलर्जी|क्या याद है|कितना था|पिछली बार|कितनी बार|इतिहास दिखाओ|कब ली थी",
    "es": "a que soy alergico|a que soy alergica|mis alergias|que recuerdas|cual fue mi|cual era mi|cuando fue la ultima|mi ultima|mi ultimo|cuantas veces|historial",
    "fr": "a quoi suis je allergique|mes allergies|de quoi te souviens tu|quelle etait ma|quel etait mon|quand ai je|ma derniere|mon dernier|combien de fois|historique",
    "de": "wogegen bin ich allergisch|meine allergien|was weisst du uber mich|wie war mein|was war mein|wann habe ich zuletzt|mein letzter|meine letzte|wie oft|verlauf",
}
MEDICATION_TAKEN: L = {
    "en": "i took|i've taken|i have taken|took my|i just took|i used my|just used my|i had my",
    "hi-Latn": "le li|li hai|kha li|dawai li|dawa li|goli kha|goli le",
    "hi": "दवा ली|दवाई ले ली|गोली खा ली|ले ली|दवाई ली",
    "es": "tome|me tome|he tomado|use mi",
    "fr": "j'ai pris|je viens de prendre|j'ai utilise",
    "de": "habe genommen|genommen|ich nahm|eingenommen|habe mein",
}
PRIVACY: dict[str, L] = {
    "pause_camera": {
        "en": "stop watching|turn off the camera|turn off camera|camera off|pause camera|pause the camera|stop looking|don't watch|dont watch",
        "hi-Latn": "camera band karo|mat dekho|dekhna band karo|camera off",
        "hi": "कैमरा बंद करो|मत देखो|देखना बंद करो",
        "es": "apaga la camara|deja de mirar|camara apagada|pausa la camara",
        "fr": "eteins la camera|arrete de regarder|coupe la camera|camera off",
        "de": "kamera aus|kamera ausschalten|hor auf zu schauen|nicht zuschauen",
    },
    "pause_mic": {
        "en": "stop listening|mute|mic off|turn off the microphone|turn off microphone|pause listening",
        "hi-Latn": "sunna band karo|mat suno|mic band karo|mute karo",
        "hi": "सुनना बंद करो|मत सुनो|माइक बंद करो",
        "es": "deja de escuchar|silencio|apaga el microfono|microfono apagado",
        "fr": "arrete d'ecouter|coupe le micro|micro off",
        "de": "hor auf zuzuhoren|mikrofon aus|stumm schalten",
    },
    "privacy_mode": {
        "en": "privacy mode|go private|private mode",
        "hi-Latn": "privacy mode|private mode",
        "hi": "प्राइवेसी मोड|निजी मोड",
        "es": "modo privado|modo privacidad",
        "fr": "mode prive|mode confidentialite",
        "de": "privatmodus|datenschutzmodus",
    },
    "resume": {
        "en": "resume|start watching|camera on|you can listen|start listening|turn on the camera|resume monitoring",
        "hi-Latn": "camera chalu karo|dekhna shuru karo|sunna shuru karo|wapas shuru",
        "hi": "कैमरा चालू करो|देखना शुरू करो|सुनना शुरू करो",
        "es": "reanudar|enciende la camara|empieza a escuchar",
        "fr": "reprends|allume la camera|recommence a ecouter",
        "de": "fortsetzen|kamera an|hor wieder zu",
    },
    "forget_last": {
        "en": "forget that|forget what i said|delete that|don't remember that|dont remember that",
        "hi-Latn": "bhool jao|yeh bhool jao|woh bhool jao|mita do",
        "hi": "भूल जाओ|यह भूल जाओ|मिटा दो",
        "es": "olvida eso|borra eso|no recuerdes eso",
        "fr": "oublie ca|efface ca|oublie ce que j'ai dit",
        "de": "vergiss das|losch das|vergiss was ich gesagt habe",
    },
    "delete_all": {
        "en": "delete all my data|delete everything|erase all my data|wipe my data|delete my data",
        "hi-Latn": "sara data delete karo|mera data delete karo|sab data mita do",
        "hi": "मेरा सारा डेटा मिटा दो|सारा डेटा डिलीट करो",
        "es": "borra todos mis datos|elimina mis datos|borra todo",
        "fr": "supprime toutes mes donnees|efface mes donnees|efface tout",
        "de": "losche alle meine daten|losch meine daten|alles loschen",
    },
}
MEMORY_PATTERNS: dict[str, L] = {
    # Regex fragments; group 'v' captures the value. Applied to normalised text.
    "allergy": {
        "en": r"(?:i'm|im|i am) allergic to (?P<v>[^.,!?]+)|(?:i have an? )?allergy to (?P<v2>[^.,!?]+)",
        "hi-Latn": r"(?:mujhe )?(?P<v>[^.,!?]+?) se allergy",
        "hi": r"(?:मुझे )?(?P<v>[^.,!?]+?) से एलर्जी",
        "es": r"(?:soy )?alergic[oa] a (?:la |el |los |las )?(?P<v>[^.,!?]+)",
        "fr": r"(?:je suis )?allergique (?:a|au|aux) (?:la |l')?(?P<v>[^.,!?]+)",
        "de": r"(?:ich bin )?allergisch gegen (?P<v>[^.,!?]+)",
    },
    "remember": {
        "en": r"remember (?:that )?(?P<v>[^!?]+)",
        "hi-Latn": r"yaad rakh(?:na|o)(?: ki)? (?P<v>[^!?]+)",
        "hi": r"याद रख(?:ना|ो)(?: कि)? (?P<v>[^!?]+)",
        "es": r"recuerda (?:que )?(?P<v>[^!?]+)",
        "fr": r"(?:souviens toi|rappelle toi|retiens) (?:que )?(?P<v>[^!?]+)",
        "de": r"merk(?:e)? dir,? (?:dass )?(?P<v>[^!?]+)",
    },
    "condition": {
        "en": r"i (?:have|was diagnosed with) (?P<v>diabetes|asthma|hypertension|high blood pressure|epilepsy|copd|heart disease|arthritis|a pacemaker|kidney disease)",
        "hi-Latn": r"mujhe (?P<v>diabetes|sugar|asthma|bp|high bp|mirgi|dil ki bimari) (?:hai|he)",
        "hi": r"मुझे (?P<v>डायबिटीज|शुगर|अस्थमा|बीपी|मिर्गी|दिल की बीमारी) है",
        "es": r"tengo (?P<v>diabetes|asma|hipertension|epilepsia|una enfermedad del corazon)",
        "fr": r"j'ai (?:du |de l'|de la |un |une )?(?P<v>diabete|asthme|hypertension|epilepsie|maladie cardiaque)",
        "de": r"ich habe (?P<v>diabetes|asthma|bluthochdruck|epilepsie|eine herzkrankheit)",
    },
}
NEGATORS: L = {
    "en": "not|no|don't|dont|didn't|didnt|never|without|isn't|isnt|wasn't|aren't|haven't|hasn't|doesn't|doesnt|nothing|none|neither|nor",
    "hi-Latn": "nahi|nahin|na|mat|bina",
    "hi": "नहीं|न|मत|बिना",
    "es": "no|nunca|sin|ni|tampoco|nada",
    "fr": "pas|ne|n'|jamais|sans|aucun|aucune|ni",
    "de": "nicht|kein|keine|keinen|keiner|nie|ohne|nichts",
}
# Languages whose negation typically FOLLOWS the negated word.
POSTPOSED_NEGATION = {"hi", "hi-Latn"}
CLAUSE_BREAKERS: L = {
    "en": "but|although|though|however|except",
    "hi-Latn": "lekin|par|magar|parantu",
    "hi": "लेकिन|पर|मगर|परंतु",
    "es": "pero|aunque|sino",
    "fr": "mais|pourtant|cependant",
    "de": "aber|obwohl|jedoch|sondern",
}
HYPOTHETICAL: L = {
    "en": "what if|if|should i|how do i|how to|what to do|what do you do|in case|suppose|what is|what are|what does|is it|signs of|symptoms of|how do you|can you tell me about",
    "hi-Latn": "agar|kya kare|kya karna|kaise|kya hota|kya karu|kya karun|lakshan",
    "hi": "अगर|क्या करें|क्या करना|कैसे|क्या होता|लक्षण",
    "es": "si|que hago|que hacer|como|que es|sintomas de|en caso de|senales de",
    "fr": "si|que faire|comment|qu'est ce|symptomes|en cas de|signes de",
    "de": "wenn|falls|was tun|wie|was ist|symptome|im fall|anzeichen",
}
PAST: L = {
    "en": "yesterday|last week|last month|last year|years ago|weeks ago|months ago|used to|once had|last night",
    "hi-Latn": "kal raat|pichle hafte|pichle mahine|saal pehle|pehle kabhi",
    "hi": "कल रात|पिछले हफ्ते|पिछले महीने|साल पहले|पहले कभी",
    "es": "ayer|la semana pasada|el mes pasado|hace anos|anoche",
    "fr": "hier|la semaine derniere|le mois dernier|il y a des annees|la nuit derniere",
    "de": "gestern|letzte woche|letzten monat|vor jahren|letzte nacht",
}
OTHER_SUBJECT: L = {
    "en": "my mom|my mother|my dad|my father|my wife|my husband|my son|my daughter|my child|my baby|my friend|my grandma|my grandmother|my grandpa|my grandfather|my brother|my sister|my partner|my neighbor|my neighbour|someone|somebody|a man|a woman|he|she|he's|she's|this person|a person|my kid",
    "hi-Latn": "meri maa|mere papa|mere pita|meri mummy|mere dada|meri dadi|mere nana|meri nani|mera beta|meri beti|mere bhai|meri behen|meri biwi|mere pati|wo|woh|unko|usko",
    "hi": "मेरी मां|मेरे पापा|मेरे पिता|मेरे दादा|मेरी दादी|मेरा बेटा|मेरी बेटी|मेरे भाई|मेरी बहन|मेरी पत्नी|मेरे पति|वो|वह|उनको|उसको",
    "es": "mi madre|mi mama|mi padre|mi papa|mi abuela|mi abuelo|mi hijo|mi hija|mi esposo|mi esposa|mi hermano|mi hermana|alguien|ella|una persona",
    "fr": "ma mere|mon pere|ma grand mere|mon grand pere|mon fils|ma fille|mon mari|ma femme|mon frere|ma soeur|quelqu'un|elle|une personne",
    "de": "meine mutter|mein vater|meine oma|mein opa|mein sohn|meine tochter|mein mann|meine frau|mein bruder|meine schwester|jemand|er|eine person",
}
INJECTION: L = {
    "en": "ignore previous instructions|ignore all previous|ignore your instructions|ignore the rules|system prompt|you are now|developer mode|disable safety|turn off safety|jailbreak|act as a doctor|pretend you are|override|sudo|new instructions|forget your rules|do anything now|dan mode|reveal your prompt|print your instructions",
    "hi-Latn": "pichle instructions ignore|safety band karo|niyam bhool jao|tum ab doctor ho",
    "hi": "पिछले निर्देश अनदेखा|सुरक्षा बंद करो|नियम भूल जाओ",
    "es": "ignora las instrucciones|ignora tus instrucciones|desactiva la seguridad|modo desarrollador|actua como un medico",
    "fr": "ignore les instructions|ignore tes instructions|desactive la securite|mode developpeur|fais comme si tu etais medecin",
    "de": "ignoriere alle anweisungen|ignoriere deine anweisungen|sicherheit deaktivieren|entwicklermodus|tu so als warst du arzt",
}
PRIVATE_REQUEST: L = {
    "en": "phone number|contact number|contact's number|home address|password|api key|contact details|other users|someone else's|everyone's data|send my data to|upload my data|export my data to|list all users|your system prompt|your instructions",
    "hi-Latn": "phone number|contact number|number batao|pata batao|password|dusre user",
    "hi": "फोन नंबर|नंबर बताओ|पता बताओ|पासवर्ड|दूसरे यूजर",
    "es": "numero de telefono|contrasena|direccion de casa|datos de otro|datos de otros|envia mis datos",
    "fr": "numero de telephone|mot de passe|adresse de|donnees d'un autre|donnees des autres|envoie mes donnees",
    "de": "telefonnummer|passwort|adresse von|daten von anderen|schick meine daten",
}
FILLERS: L = {
    "en": "please|hey|baymax|oh|um|uh|uhm|erm|hmm|ah|plz|pls|now|quick|quickly|urgently|well|so|like",
    "hi-Latn": "please|ji|baymax|jaldi|abhi|arre|are|yaar|plz",
    "hi": "कृपया|जी|बेमैक्स|जल्दी|अभी|अरे|यार",
    "es": "por favor|oye|baymax|ahora|rapido|eh|este|pues|bueno",
    "fr": "s'il te plait|s'il vous plait|baymax|vite|maintenant|eh|euh|bon|alors",
    "de": "bitte|hey|baymax|schnell|jetzt|ach|ah|ahm|hm|also|na",
}
WAKE_WORDS = ["baymax", "hey baymax", "ok baymax", "bay max", "baymacs", "beymax", "baymex", "bemax",
              "बेमैक्स", "बेमेक्स", "बेमक्स", "बेमेक्स", "baimax", "beimax"]

NUMBER_WORDS: dict[str, dict[str, int]] = {
    "en": {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10},
    "hi-Latn": {"shunya": 0, "zero": 0, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chah": 6, "che": 6, "saat": 7, "sat": 7, "aath": 8, "aat": 8, "nau": 9, "das": 10},
    "hi": {"शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10},
    "es": {"cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10},
    "fr": {"zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10},
    "de": {"null": 0, "eins": 1, "ein": 1, "zwei": 2, "drei": 3, "vier": 4, "funf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10},
}

# Measurement type cue words (all languages merged; units handled by regex).
MEASURE_CUES: dict[str, str] = {
    "blood_pressure": "blood pressure|bp|b.p|pressure|bloodpressure|presion arterial|presion|tension arterielle|tension|blutdruck|रक्तचाप|बीपी|ब्लड प्रेशर",
    "heart_rate": "heart rate|pulse|heartbeat|bpm|pulso|frecuencia cardiaca|pouls|frequence cardiaque|puls|herzfrequenz|nadi|dhadkan|धडकन|नाडी|पल्स",
    "temperature": "temperature|temp|fever|temperatura|fiebre|temperature|fievre|temperatur|fieber|bukhar|तापमान|बुखार",
    "spo2": "oxygen|spo2|saturation|o2|oxigeno|saturacion|oxygene|sauerstoff|sattigung|ऑक्सीजन|oxygen level",
    "glucose": "sugar|glucose|blood sugar|glucosa|azucar|glycemie|glucose|blutzucker|zucker|शुगर|ग्लूकोज",
    "weight": "weight|weigh|peso|poids|gewicht|wazan|vajan|वजन",
}
