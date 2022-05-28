# TREBALL DE FI DE GRAU 2021-2022
### Nom: David Sardà Martin
### DATASET: Detecting Obfuscated Malware using Memory Feature Engineering.
### URL: https://www.unb.ca/cic/datasets/malmem-2022.html 
## Resum
El dataset escollit és un dataset publicat per la Universitat de New Brunswick, anomenat CIC-MalMem-2022.

Per la creació del Dataset es va fer a partir de recollir mostres benignes, les mostres benignes són extretes a partir d'executar diverses aplicacions en una màquina simulant el comportament d'un usuari mitjà, i malignes, extretes de 2916 mostres de virus de VirusTotal amb diferents categories, entre les quals s'inclou Ransomware que serà la que ens interessa pel nostre projecte, en el mateix tipus de dispositiu.
El dataset fa un recull de diferents paràmetres de memòria per tal de veure si els atacs de malware ofuscat es poden detectar a partir de la memòria. Pels nostres objectius d'aquest projecte eliminarem les mostres que no corresponen a mostres benignes o d'atacs de Ransomware.

Tenim un total de 58596 instàncies o mostres (corresponents a les diferents files) i 57 atributs, corresponen als diferents paràmetres de memòria recol·lectats, el tipus de virus que s'ha provat i si es benigne o maligne. Excepte aquests dos últims atributs tots corresponen a dades numèriques. D'aquestes la mostra de malware provada serà eliminada 'Category' i l'atribut 'Class' serà convertida en numeral.



## Objectius del dataset

Amb aquest Dataset volem crear un model capaç de predir si donades certes característiques sobre la memòria d'un dispositiu, poder predir si aquest 
dispositiu es troba sota un atac de Malware de tipus Ransomware. D'aquesta forma l'objectiu és la detecció d'atacs de Ransomware tenint en compte la importància
de predir aquests tipus d'atac amb precisió, que no es produeixin falsos negatius o falsos positius i amb rapidesa per tal de posteriorment poder executar una resposta 
de forma immediata per evitar qualsevol conseqüència de l'atac.

Per la selecció de models finals per predir detectar Ransomware es crearà un model en forma de xarxa neuronal amb PyTorch un model simple i un de complex.


## Organització del Github

- data: En aquest directori hi ha el dataset a partir del qual intentarem predir si un dispositiu esta sota un atac de Ransomware.
- demo: En aquest directori trobem el fitxer que ens mostra com podem executar el model creat.
- figures: En aquest directori trobem les diferents imatges que s'han fet per analitzar el dataset i el resultat dels models.
- models: En aquest directori trobem guardats els models creats.
- notebook: En aquest directori trobem guardar un fitxer jupyter noteboook que mostra els resultats obtinguts.
- results: En aquest directori trobem el resultat de les execucions on imprimiem els diferents resultats o dades que s'han considerat necessàries.
- source: En aquest directori hi ha el procès de creació dels diferents models creats així com el anàlisi d'ells.
  - main.py: En aquest fitxer hi ha procès de creació dels models creats amb la llibreria sklearn.
  - xarxa neuronal.py: En aquest fitxer hi ha les proves fetes per entendre el funcionament i procès de crear un model de xarxa neuronal amb PyTorch.
  - xarxa_neuronal_ransom.py: En aquest fitxer hi ha el procès de creació del model de xarxa neuronal amb PyTorch pel dataset de Ransomware.
  - modelsfinals.py: En aquest fitxer guardem els models finals sel·leccionats.
- README.md: Aquest document explicar el contingut del Github.
- requeriments.txt: En aquest fitxer podem trobar les diferents llibreries que s'han instal·lat per tal de poder executar el codi.

## Experiments

En aquesta pràctica hem provat diferents models de classificació, posteriorment per a millorar més el model buscarem els millors hiperparàmetres pel model 
i finalment obtindrem els millors models i hiperparàmetres segons la precisió i altres paràmetres per considerar les millors característiques. A més també especificament, provarem 
la creació d'una xarxa neuronal amb la llibreria de PyTorch provant com afecten diferents paràmetres.


## Preprocessat

En el preprocessat hem normalitzat les dades prèviament a provar qualsevol model de classificació, hem vist que no tenim atributs Nans
per això no caldrà cap conversió dels atributs Nans. 

En el preprocessat s'han eliminat algunes categories ja que no aportaven informació addicional a la resta de categories, ja que aportaven la mateixa que altres o no n'aportaven cap
que són les següents: 'pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services','callbacks.ngeneric', 'callbacks.nanonymous', 'pslist.avg_handlers', 'ldrmodules.not_in_mem', 'ldrmodules.not_in_load_avg',
'malfind.protection', 'psxview.not_in_pslist',  'psxview.not_in_session_false_avg','psxview.not_in_csrss_handles' i 'Category'. Eliminar aquestes categories ens permetrà que els algoritmes convergeixin més ràpid i també fagin predicció més ràpidament.




## Model

En aquesta secció, veurem els resultats al provar diferents models de la llibreria sklearn segons diferents mètriques seleccionades:


Model|accuracy (desviació típica)|f1 score|recall|roc_auc|temps convergir (s) |temps test (s) |
| ----------- | ------------------------------------------------- | -------- | -------- | ------ | ------- | ------- |
SVM rbf|0.999420 (0.000168)|0.999437|0.999437|0.999997|3.952055|0.129571|
SVM sigmoide|0.500000 (0.000000)|0.642215|0.749520|0.981608|317.390107|11.033297|
SVM polinomi|0.999949 (0.000078)|0.999923|0.999923|1.000000|1.234851|0.021345|
SVM linear|0.999086 (0.000500)|0.999182|0.999181|0.999934|3.477355|0.071629|
Logistic Regression|0.998994 (0.000353)|0.999105|0.999105|0.999891|0.705057|0.023502|
Guassian Naive Bayes|0.994591 (0.000422)|0.991933|0.991890|0.996977|0.037387|0.019007|
Linear Discriminant Analysis|0.996694 (0.000599)|0.997799|0.997800|0.999063|0.248959|0.024935|
Decision Tree|0.999813 (0.000259)|0.999872|0.999872|0.999813|0.160248|0.011418|
K Nearest Neighbors|0.999743 (0.000246)|0.999821|0.999821|0.999933|0.018677|8.562271|
Extra Trees|0.999933 (0.000110)|0.999949|0.999949|1.000000|0.883781|0.072164|
Random Forest|0.999915 (0.000149)|0.999923|0.999923|0.999982|1.163956|0.442206|
HistGradientBoosting|0.999966 (0.000076)|0.999949|0.999949|0.999981|0.782509|0.048980|
ADABoosting|0.999914 (0.000151)|0.999923|0.999923|1.000000|9.531644|0.232996|
Bagging Classifier|0.994505 (0.000528)|0.991807|0.991762|0.997664|0.375173|0.077892|
GradientBoostingClassifier|0.999914 (0.000111)|0.999923|0.999923|0.999965|6.528635|0.016349|


També s'ha creat una xarxa neuronal amb PyTorch.

## OPTIMITZACIÓ

S'han seleccionat aquells que ofereixen un millor rendiment segons les mètriques triades, particularment s'ha prioritzat un equilibri entre 
el temps de les mostres del dataset per predir ràpidament si s'esta produïnt i l'accuracy i recall per detectar i classificar correctament si es tracta d'un
atac de Ransomware o no. A continuació podem veure els models seleccionats i el seu rendiment un cop optimitzats, juntament amb el rendiment de la
xarxa neuronal creada amb PyTorch:

Model |accuracy (desviació típica)|f1 score|recall|roc_auc|temps convergir (s) |temps test (s) |
| ----------- | ------------------------------------------------- | -------- | -------- | ------ | ------- | ------- |
HistGradientBoosting optimitzat|0.999983 (0.000038)|0.999974|0.999974|0.999999|0.692764|0.047505|
SVM polinomi optimitzat|0.999932 (0.000113)|0.999949|0.999949|1.000000|1.459770|0.024470|
Xarxa Neuronal pròpia|0.99136 ( - ) ||||81.1464| 0.0114|


L'optimització es va fer amb l'algoritme Randomized Search pels algoritmes HistGradientBoosting i SVM polinomial. Per la xarxa neuronal l'optimitzador que s'ha seleccionat per 
actualitzar els pesos ha sigut el SGD.

## Demo

Podem provar d'executar una demo

## Comparació de Resultats

No hi ha cap predicció que s’hagi fet exactament amb el mateix Dataset, ja que s’ha adaptat pel nostre cas en concret, extraient del Dataset tots aquells casos que no corresponien a mostres benignes o d’atacs de Ransomware. Per tan la comparació no es farà especificament pel dataset amb el que hem treballat.


Si comparem amb els resultats obtinguts pels creadors del Dataset [1], podem veure que els resultats dels algoritmes es lleugerament superior pel nostre cas de predir Ransomware que en el seu cas per predir malware ofuscat i obtenen resultats de fins a 98% en precisió i 97% de recall.




D’aquesta forma, podem pensar que els nostres models funcionen correctament i les diverses decisions fetes per preprocessar el dataset també estan funcionen correctament i inclús superen lleugerament els models d’aquest estudi.


Al llarg del temps s’han fet nombrosos estudis per predir atacs de Ransomware amb diferents datasets, un altre exemple es l’estudi realitzat [2] en el que es crea un dataset per predir atacs de Ransomware amb anàlisi dinàmic, en aquest projecte els resultats experimentals mostren el mètode proposat pot detectar atacs de ransomware utilitzant només característiques de comportament de baix nivell. De forma que si els atacants poden comprometre la primera capa de protecció, la capa de seguretat addicional seria útil.

Brengel va presentar un mètode de detecció de virtualització basat en el temps mitjançant la sobrecàrrega de sortida de VM i la memòria intermèdia de traducció assolint un 95.95% de tassa d'encert.


Altres exemples d’estudis realitzats són: 

 - “Ransomware Detection Using the Dynamic Analysis and Machine Learning: A Survey and Research Directions" [3], 
 - "Ransomware Detection Using Machine Learning" [4], 
 - "Ransomware Detection using Random Forest Technique", on es mostra que amb el Random Forest es va aconseguir una accuracy de fins a 97.74% d'encert. També es mostra diversos estudis de predicció de Ransomware a partir de trucades a l'API que es representen a partir de vectors q-gram, on els resultats amb models SVM mostren precisió de 97.48%. Vinayakumar va proposar mètodes per la detecció d'atcs de Ransomware a partir de recollir les seqüènciaes de l'API amb anàlisi dinàmic, amb un model multicapa de perceptró (MLP) van aconseguir una precisió del 98%. Homayoun va introduir un sistema de detecció de ransomware basat en la mineria de patrons seqüencials com a característiques candidates per utilitzar-les com a entrada a les tècniques d'aprenentatge computacional (MLP, Bagging, Random Forest i J48) amb finalitats de classificació. Els resultats van mostrar una precisió de fins el 99% per a la detecció de ransomware.[5] 
 - "API-Based Ransomware Detection Using Machine Learning-Based Threat Detection Models". Aquesta anàlisi va obtenir una alta precisió de detecció de ransomware del 99,18% per a plataformes basades en Windows i mostra el potencial d'aconseguir capacitats de detecció de ransomware d'alta precisió quan s'utilitza una combinació de trucades d'API i un model ML. [6] 

D'aquesta forma comparat amb els nostres resultats, podem veure que hem acosneguit crear models que ofereixen un molt bon rendiment, amb un rendiment molt similars als millors que s'han creat, ja que de fet és un model que quasi no comet errors i a més a partir de les dades es capaç de detectar si és un atac de Ransomware molt ràpidament.
## Conclusions

Els atacs de Ransomware poden tenir conseqüències molt greus per les víctimes, ja siguin persones o institucions, poden bloquejar l’accés a dispositius, robar dades o bloquejar tots els serveis informàtics d’una organització. El que pot comportar grans pèrdues econòmiques o del dret a la privacitat. Els atacs de Ransomware són del tipus malware ofuscat, és a dir que es mantenen ocults en el sistema sense que l’usuari se n’adoni, addicionalment això són atacs que al llarg del temps evolucionen i segueixen nous patrons de forma que són molt complicats de detectar. A causa d’això, van aparèixer nombrosos estudis per detectar aquest tipus d’atac per poder posteriorment respondre’n. En aquest projecte s’han creat models per detectar atacs de Ransomware.

Al final del projecte hem vist que els diferents objectius marcats s’han assolit satisfactòriament. S’han creat un model SVM i un model HistoGradientBoosting per predir atacs de Ransomware amb un percentatge d’encert superior al 99%, a més també s’ha creat una Xarxa Neuronal amb dues capes ocultes que ens donava també una tassa d’encert similar, i a més ens oferia millores pel que fa al temps de predicció. Per realitzar el treball, s’ha entès amb més aprofundiment el funcionament, els algorismes i paràmetres del Machine Learning i com treballar amb ells. 



## Idees per treballar en un futur

De cara el futur es poden utilitzar els models d'aquest Github per crear un script que reculli els paràmetres i detecti si hi ha Ransomware i posteriorment i actui d'acord al resultat, en cas positiu protegint algun element del dispositiu. També, es pot
provar de crear una xarxa convolucional amb PyTorch i veure el resultat que ofereix segons les mètriques que hem vist.

## BIBLIOGRAFIA
[1]  https://www.scitepress.org/Papers/2022/109082/109082.pdf

[2]  https://www.sciencedirect.com/science/article/pii/S2666281721002390.

[3]  https://doi.org/10.3390/app12010172

[4]  https://spinbackup.com/blog/ransomware-detection-using-machine-learning

[5]  https://www.sciencedirect.com/science/article/pii/S2405959520304756.

[6]  https://ieeexplore.ieee.org/document/9647816

## Llicencia
UAB