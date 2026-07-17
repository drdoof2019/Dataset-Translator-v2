# 🌐 Dataset Translator — Kullanım Kılavuzu

HuggingFace veri setlerini yerel bir LLM (LM Studio) kullanarak çeviren bir Gradio web uygulamasıdır.

---

## 📋 İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Kurulum](#kurulum)
3. [LM Studio Ayarları](#lm-studio-ayarları)
4. [Uygulamayı Başlatma](#uygulamayı-başlatma)
5. [Sekmeler ve Kullanım](#sekmeler-ve-kullanım)
   - [⚙️ Yapılandırma (Config)](#️-yapılandırma-config)
   - [📥 İndirme (Download)](#-indirme-download)
   - [🌐 Çeviri (Translate)](#-çeviri-translate)
   - [📤 Yükleme (Upload)](#-yükleme-upload)
6. [JSON Hücre Desteği](#json-hücre-desteği)
7. [Devam Etme (Resume) Mekanizması](#devam-etme-resume-mekanizması)
8. [Dosya Yapısı](#dosya-yapısı)
9. [Sorun Giderme](#sorun-giderme)

---

## Gereksinimler

- **Python 3.10+**
- **LM Studio** — Yerel LLM sunucusu (https://lmstudio.ai)
- **TranslateGemma** modeli (veya tercih ettiğiniz çeviri modeli)
- İnternet bağlantısı (veri seti indirme ve HuggingFace yükleme için)

---

## Kurulum

1. **Sanal ortamı oluşturun** (ilk kurulumda):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **`.env` dosyasını düzenleyin**:
   ```
   LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
   LM_STUDIO_MODEL=translategemma-12b-it
   LM_STUDIO_API_KEY=lm-studio
   HF_TOKEN=hf_xxxxxxx
   DEFAULT_SOURCE_LANG=en
   DEFAULT_TARGET_LANG=tr
   ```

   | Değişken | Açıklama |
   |---|---|
   | `LM_STUDIO_BASE_URL` | LM Studio sunucu adresi |
   | `LM_STUDIO_MODEL` | LM Studio'da yüklü model adı |
   | `LM_STUDIO_API_KEY` | API anahtarı (LM Studio varsayılan: `lm-studio`) |
   | `HF_TOKEN` | HuggingFace yazma erişim anahtarı |
   | `DEFAULT_SOURCE_LANG` | Varsayılan kaynak dil kodu (ör: `en`) |
   | `DEFAULT_TARGET_LANG` | Varsayılan hedef dil kodu (ör: `tr`) |

---

## LM Studio Ayarları

1. **LM Studio'yu açın** ve sol panelden model indirin (ör: `translategemma-12b-it`).
2. Modeli yükleyin (Local Server sekmesi → model seçin → "Start Server").
3. Varsayılan adres: `http://127.0.0.1:1234/v1`
4. Uygulama içindeki **⚙️ Config** sekmesinden de adres ve model adını değiştirebilirsiniz.

> **İpucu:** LM Studio'da model yüklü ve sunucu çalışıyorken "Test LM Studio Connection" butonu ile bağlantıyı doğrulayın.

---

## Uygulamayı Başlatma

```powershell
.venv\Scripts\activate
python app.py
```

Tarayıcınızda `http://localhost:7860` adresini açın.

---

## Sekmeler ve Kullanım

### ⚙️ Yapılandırma (Config)

Bu sekme temel ayarları yapılandırmanızı sağlar:

- **LM Studio Base URL** — Yerel LLM sunucu adresi
- **Model Name** — LM Studio'da yüklü modelin adı
- **🔌 Test LM Studio Connection** — LM Studio bağlantısını test eder
- **HF Token** — HuggingFace Hub yazma erişim anahtarı
- **🔑 Test HF Token** — HuggingFace token doğrulaması yapar
- **Source / Target Language** — Varsayılan kaynak ve hedef diller

### 📥 İndirme (Download)

HuggingFace'ten veri seti indirir ve yerel olarak kaydeder.

1. **Dataset ID** alanına veri seti kimliğini girin:
   - Örnek: `WithinUsAI/claude_mythos_distilled_25k`
2. **🔍 Fetch Splits** butonuna tıklayın — Veri setinin mevcut split'lerini otomatik algılar ve Split açılır menüsünü günceller.
3. **Split** — Veri seti bölmesi (Fetch Splits ile otomatik doldurulur, manuel de değiştirilebilir).
4. **Row Limit** — İndirilecek satır sayısı (`0` = tamamı, varsayılan `0`).
   - Örnek: `1000` yazarsanız ilk 1000 satır indirilir.
5. **⬇️ Download Dataset** butonuna tıklayın.
6. İndirilen veri setinin **önizlemesi** (ilk 10 satır) görüntülenir.

#### 🔍 Hücre İnceleyici (Cell Inspector)

Önizlemede herhangi bir hücreye tıkladığınızda, içeriği aşağıdaki "Cell Content" alanında **pretty JSON** formatında görüntülenir. Uzun JSON/array içeriklerini rahatça okuyabilirsiniz.

#### 📊 Sütun Analizi

İndirme sonrası her sütunun JSON yapısı otomatik analiz edilir:
- 🟢 **Çevrilecek anahtarlar** — JSON Translate Keys listesinde bulunanlar
- 🔴 **Atlanacak anahtarlar** — JSON Skip Keys listesinde bulunanlar
- ⚪ **Dokunulmayacak anahtarlar** — Her iki listede de olmayanlar

#### Sütun Seçimi

7. **📝 Select Text Columns** — Metin (object tipi) sütunları otomatik seçer.
8. **✅ Select All** — Tüm sütunları seçer.
9. **🗑️ Clear Selection** — Tüm seçimleri temizler.
10. Seçim altındaki **📌 N column(s) selected** yazısı, kaç sütun seçili olduğunu canlı olarak gösterir.

> **Önbellek:** Aynı veri seti daha önce indirildiyse, tekrar indirilmez; yerel dosyadan yüklenir.

### 🌐 Çeviri (Translate)

Seçili sütunları hücre hücre çevirir.

1. **Translation Settings** bölümünde çeviri bilgileri görüntülenir.
2. **JSON Hücre İşleme** bölümünde JSON modu seçilir (aşağıda detay).
3. **▶️ Start Translation** — Çeviriyi başlatır.
4. **⏹️ Stop** — Mevcut satır bittikten sonra çeviriyi durdurur.
5. **🗑️ Clear Progress** — Kaydedilen ilerlemeyi sıfırlar (baştan başlar).

**İlerleme çubuğu** ve **durum mesajı** her 2 saniyede bir otomatik güncellenir.

### 📤 Yükleme (Upload)

Çevrilmiş veri setini HuggingFace Hub'a yükler.

1. **Translated CSV Path** — Çeviri tamamlandığında otomatik doldurulur.
2. **Repo Name** — HuggingFace'te oluşturulacak repo adı.
3. **Private** — Depoyu gizli yapar.
4. **📄 Generate README Preview** — Veri seti için otomatik README.md oluşturur.
5. **🚀 Upload to HuggingFace** — Veri setini ve README'yi yükler.

---

## JSON Hücre Desteği

Bazı veri setlerinde hücreler JSON yapısında olabilir (ör: çok turlu konuşma verileri). Bu araç bu tür hücreleri akıllıca işler.

**Örnek JSON hücre:**
```json
[
  {"role": "user", "content": "What is AI?"},
  {"role": "assistant", "content": "AI is artificial intelligence..."}
]
```

### JSON Modları

| Mod | Açıklama |
|---|---|
| **auto** (varsayılan) | Hücrenin JSON olup olmadığını otomatik algılar. JSON ise `translate_json()` kullanır, değilse `translate_cell()` kullanır. |
| **force** | Tüm hücreleri JSON olarak işler. JSON olmayan hücreler normal çeviriye düşer. |
| **off** | JSON algılama devre dışı. Tüm hücreler düz metin olarak çevrilir. |

### Translate Keys (Çevrilecek Anahtarlar)

JSON yapısında hangi anahtarların değerlerinin çevrileceğini belirler. Varsayılan:

```
content, text, summary, description, question, answer, input, output, response, message, instruction, completion
```

### Skip Keys (Atlanacak Anahtarlar)

JSON yapısında hiçbir dokunulmayacak anahtarlar. Varsayılan:

```
role, id, name, type, model, source, lang, language, timestamp, created_at, updated_at, index, metadata
```

> **İpucu:** Farklı JSON yapısına sahip veri setleri için Translate Keys ve Skip Keys listesini uygulama içinden düzenleyebilirsiniz.

---

## Devam Etme (Resume) Mekanizması

Çeviri herhangi bir nedenle kesilirse (kapanma, hata, manuel durdurma):

1. **İlerleme otomatik kaydedilir** — Her satır işlendikten sonra `output/state/` klasörüne checkpoint yazılır.
2. **Tekrar başlatıldığında** — Aynı veri seti, split ve sütunlar seçilip "Start Translation" tıklanırsa, **kaldığı yerden devam eder**.
3. **Baştan başlamak için** — "🗑️ Clear Progress" butonunu kullanın.

> **Not:** Checkpoint dosyası `output/state/` klasöründe `<dataset_id>_<split>_<columns_hash>.json` formatında saklanır.

---

## Dosya Yapısı

```
dataset_translation_v2/
├── .env                      # Ortam değişkenleri (token, LM Studio ayarları)
├── .gitignore                # Git ignore kuralları
├── app.py                    # Gradio web uygulaması (ana giriş noktası)
├── config.py                 # Yapılandırma yükleme
├── translator.py             # LLM çeviri sınıfı (LM Studio)
├── requirements.txt          # Python bağımlılıkları
├── KULLANIM.md               # Bu dosya
├── core/
│   ├── __init__.py
│   ├── dataset_loader.py     # HuggingFace veri seti indirme/kaydetme
│   ├── progress_manager.py   # İlerleme takibi ve checkpoint
│   ├── translation_engine.py # Hücre hücre çeviri motoru
│   └── hf_uploader.py        # HuggingFace Hub yükleme
├── output/
│   ├── datasets/             # İndirilen veri setleri (CSV + Parquet)
│   ├── translated/           # Çevrilmiş veri setleri
│   └── state/                # Checkpoint dosyaları
└── legacy/                   # Eski CLI scriptleri (referans)
    ├── main.py
    ├── huggingface_uploader.py
    ├── check_dataset.py
    └── readme_generator.py
```

---

## Sorun Giderme

### LM Studio bağlantı hatası
- LM Studio'nun çalıştığını ve modelin yüklü olduğunu kontrol edin.
- Sunucu adresinin doğru olduğundan emin olun (varsayılan: `http://127.0.0.1:1234`).
- Config sekmesindeki "Test LM Studio Connection" butonunu kullanın.

### HuggingFace yükleme hatası
- Token'ınızın **write** erişimine sahip olduğundan emin olun.
- Config sekmesindeki "Test HF Token" butonu ile doğrulayın.
- Repo adının geçerli olduğundan emin olun (sadece harf, rakam, tire ve alt çizgi).

### Çeviri çok yavaş
- LM Studio'da daha küçük/faster bir model deneyin.
- Daha az sütun seçin.
- Row Limit ile önce küçük bir alt küme ile test edin.

### Bellek hatası (Memory Error)
- Row Limit kullanarak veri setini parçalara bölün.
- LM Studio model bellek ayarlarını kontrol edin.

### Çeviri kalitesi düşük
- Farklı bir model deneyin (ör: `translategemma-12b-it`, `llama-3`, `gemma-2`).
- JSON modunu kontrol edin — bazı hücreler düz metin ise `off` modunu tercih edin.

---

## Hızlı Başlangıç — Adım Adım

1. LM Studio'yu açın, modeli yükleyin, sunucuyu başlatın.
2. `.env` dosyasını düzenleyin (HF token girin).
3. Terminalde: `.venv\Scripts\activate` → `python app.py`
4. Tarayıcıda `http://localhost:7860` açılır.
5. **Config** sekmesinde bağlantıyı test edin.
6. **Download** sekmesinde veri setini indirin ve sütunları seçin.
7. **Translate** sekmesinde çeviriyi başlatın.
8. **Upload** sekmesinde HuggingFace'e yükleyin.

---

*Bu belge `dataset_translation_v2` projesi için hazırlanmıştır.*
