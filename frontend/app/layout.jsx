import "./globals.css";

export const metadata = {
  title: "ResearchAI Studio",
  description: "Automated data preprocessing pipeline",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
