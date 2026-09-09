"use client";

// Backend-rendered EDA charts (matplotlib/seaborn) arrive as base64 PNG data
// URIs in data.eda = [{title, image}]. We just show them and offer a download.
export default function EdaGallery({ data }) {
  const eda = data.eda || [];
  if (!eda.length) return null;

  function download(image, title) {
    const a = document.createElement("a");
    a.href = image;
    a.download = title.replace(/\W+/g, "_") + ".png";
    a.click();
  }

  return (
    <section>
      <p className="sectitle">EDA charts</p>
      <div className="eda-grid">
        {eda.map((c, i) => (
          <div className="chart" key={i}>
            <div className="chart-title" title={c.title}>
              {c.title}
            </div>
            <img src={c.image} alt={c.title} className="eda-img" />
            <button className="ghost eda-dl" onClick={() => download(c.image, c.title)}>
              Download PNG
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
