import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Encore Drawing Matcher",
  description: "Match handwritten sketches to master drawings and fill JSON dimensions",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <div className="min-h-screen">
          <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
            <div className="mx-auto max-w-6xl px-6 py-4">
              <h1 className="text-xl font-semibold tracking-tight">Encore Drawing Matcher</h1>
              <p className="text-sm text-slate-400 mt-1">
                Upload a handwritten sketch to match a master drawing and fill dimensions
              </p>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
