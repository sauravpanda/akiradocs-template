<p align="center">
  <h1 align="center">AkiraDocs</h1>
</p>

<p align="center">
  <a href="https://discord.gg/zvYZukgeH2">
    <img src="https://img.shields.io/badge/Join_Our_Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Join Our Discord">
  </a>
</p>

<p align="center">
  <h4 align="center"><strong>Beautiful docs that write, translate, and optimize themselves</strong></h4>
</p>

<p align="center">
  <img src="gifs/star_the_repo.gif" alt="Star the Repo" width="100%">
</p>

<p align="center">
  If you find AkiraDocs useful, please consider giving it a star! ⭐️<br>
  Your support helps us continue improving the platform.
</p>


<p align="center">
  <a href="https://demo.akiradocs.ai">
    <img src="https://img.shields.io/badge/Try_Demo-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://forms.gle/KunU4BGhToH4NJ1t7">
    <img src="https://img.shields.io/badge/Join_Waitlist_For_Cloud_Hosting-FF6B6B?style=for-the-badge&logo=googleforms&logoColor=white" alt="Join Waitlist for Cloud Hosting">
  </a>
</p>

## 🚀 What is AkiraDocs?
AkiraDocs is a modern documentation platform that combines the power of AI with a Notion-like editing experience. Create, translate, and optimize your documentation automatically while maintaining complete control over the content. Perfect for teams who want to focus on their ideas rather than the complexities of documentation management.


## 📚 WebRAG Documentation Search

AkiraDocs implements a fully client-side RAG (Retrieval Augmented Generation) system that enables unlimited AI chat interactions without any API costs or rate limits.

### How it Works

1. **Offline Indexing**: Documentation is processed and embedded during build time using GTE-small model
2. **Browser-Based Components**:
   - SQLite database (stored in browser) for vector search
   - Local embedding model for query processing
   - Local LLaMA model for response generation
   - All processing happens directly in the user's browser

### Key Benefits

- ✨ **No API Costs**: Everything runs locally in the browser
- 🚀 **Unlimited Chats**: No rate limits or usage restrictions
- 🔒 **Privacy-First**: No data leaves the user's device
- ⚡ **Low Latency**: Direct local processing after initial model load

### Technical Stack

- Embedding: `sauravpanda/gte-small-onnx`
- LLM: `Llama-3.2-1B-Instruct`
- Storage: `sql.js-httpvfs` for SQLite in browser


## ⚡️ Quick Deploy
Get started in seconds:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FCloud-Code-AI%2Fakiradocs-template)

Or via CLI:
```bash
npx create-akiradocs <optional folder name>
```

![AkiraDocs installation](gifs/install_akiradocs.gif)

## 🎯 Perfect For
- **Developers**: API docs, SDK guides, implementation examples
- **Product Teams**: User guides, release notes, tutorials
- **Enterprises**: Internal wikis, process documentation, knowledge bases

## ✨ Key Features

### Notion-like Editor
Intuitive block-based interface with real-time preview

![AkiraDocs editor](gifs/akiradocs_editor.gif)

### AI-Powered Content Generation
Automatically enhance your content with AI

![AkiraDocs AI](gifs/ai_rewrites.gif)

### Translation
Translate your content to multiple languages with AI

![AkiraDocs Translation](gifs/translate.gif)

### 🔍 AI-Generated SEO Optimization
Automatically optimize your documentation for search engines with AI-driven SEO
- Improve discoverability and reach without manual SEO adjustments
- AI recommendations for keywords and metadata

### 🛠 Developer-Friendly
- Keep your existing Markdown/Git workflow
- Full Markdown/MDX support
- Easy custom component development
- SEO optimization out of the box

### 📝 Content Creator-Friendly
- Modern block-based WYSIWYG editor
- Real-time previews
- Rich media support
- Reusable content blocks


## 🌍 Going Global
Built-in AI-powered translation. Just add API keys:
```json
{
  "translation": {
    "auto_translate": true,
    "provider": "anthropic",
    "targetLanguages": ["es", "fr", "de"]
  }
}
```

### 🤖 Upcoming Features
- **AI-Powered Documentation Generation**
  - Automatically generate comprehensive documentation from your codebase
  - AI-assisted content updates to keep your docs current with minimal effort
  - Intelligent suggestions for content improvements and expansions


## 🔄 Easy Migration (Coming Soon)
Import your existing docs:
```bash
npx akiradocs-migrate import --from=source --to=mydocs
# Supports: GitBook, Docusaurus, ReadTheDocs, Confluence, Notion
```

## 🤝 Join Our Community
- [Live Demo](https://demo.akiradocs.ai)
- [Documentation](https://docs.akiradocs.com)
- [Discord Community](https://discord.gg/zvYZukgeH2)
- [GitHub Discussions](https://github.com/Cloud-Code-AI/akiradocs/discussions)
- [Join Waitlist](https://forms.gle/KunU4BGhToH4NJ1t7) for early access

## 📄 Licensing
- Open Source (MIT License)
- Commercial License available for enterprise features

---

[Get Started](https://docs.akiradocs.ai/quickstart) •
[Live Demo](https://demo.akiradocs.ai) •
[Enterprise Trial](https://akiradocs.com/enterprise) (Coming Q1 2025)
