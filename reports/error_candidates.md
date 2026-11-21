# Error candidates

TP 126  FP 6  FN 14

## False positives (clean text flagged)
- score=0.717 [telegram_kk] У нас қалың мал
- score=0.676 [synthetic_template] Заказ келді, но түсі фотоға сәйкес емес.
- score=0.572 [synthetic_template] Заказ келді, но упаковка жыртылған, іші сау.
- score=0.553 [2gis_reviews] мынау мектеп емес түрме ғой
- score=0.525 [telegram_kk] Жалпы, ұлы далаға социализымның амбасодоры болып орыстар келді ғой
- score=0.501 [synthetic_template] Бағасы қымбат, сапасы орташа, екінші рет алмаймын.

## False negatives (profanity missed)
- score=0.088 [telegram_kk] Билігіміз орыстарға көт жалай берсін, бірақ қазақ халқы әрқашан осыны есте сақтайды
- score=0.127 [telegram_kk] Ой блә, сен прикол ұстап тұрсын ба, әлде ішіменде "кОмМуНиСтСіН" ба? Сен өзі Алаш автономиясы тұралы, голощекин "кіші қазан" саясатын (голощекинның ашаршылығы),
- score=0.156 [telegram_kk] Енді түсінікті шығар амшық
- score=0.230 [telegram_kk] Бляя ауылда үй неткен керемет
- score=0.247 [telegram_kk] Менде қотақпен әрқашан қиын болған
- score=0.285 [telegram_kk] Він до речі ебался з псом патронiм
- score=0.299 [telegram_kk] Требую до деп
- score=0.311 [telegram_kk] Ак кулактарды сыгу керек
- score=0.324 [telegram_kk] Бля жаль , қазақтар колонизациямен айналысқан жоқ , қәзір әйтпесе Оңтүстік Америкаға қазақ тілді тропикалық елге қуана-қуана көшуші едім
- score=0.327 [telegram_kk] Осындай долбоебқа өмір берген
- score=0.394 [telegram_kk] Мен қыста қалай сігілгенімді білесін ба
- score=0.437 [telegram_kk] Короче еблан болған
- score=0.445 [telegram_kk] Бургер мен че то пиздит
- score=0.485 [telegram_kk] Ажырасқаннан кейін қатындардың амын сваркілеп тастау керек, сол кезде қорықпай алимент төлей бересің👍

## Confident true positives (for contrast)
- score=1.000 [telegram_kk] Бля сука кешіктім
- score=0.999 [telegram_kk] Мал қотақ амшық гей, наху
- score=0.998 [telegram_kk] Қотақ қойдым
- score=0.998 [telegram_kk] Бля, мынау Микола қотақбас мазганы сікті уже скамить етіп
- score=0.996 [telegram_kk] Мен шығам бля
- score=0.996 [telegram_kk] Ебать ті отсталый
- score=0.995 [telegram_kk] Өзіңше пиздец
- score=0.995 [synthetic_template] Кроссовки туралы қотақбас пікір — сапасыз.
- score=0.994 [telegram_kk] Ідіть нахуй Чорнобиль топ
- score=0.994 [telegram_kk] Бля ебать мектепте отырмын және көтіммен сұйық массаны ұстап тұрмын ғой

## Mistake mix
- FP by source: {'telegram_kk': 2, '2gis_reviews': 1, 'synthetic_template': 3}
- FN by source: {'telegram_kk': 14}
- FP synthetic: 3 / 6
- FN synthetic: 0 / 14