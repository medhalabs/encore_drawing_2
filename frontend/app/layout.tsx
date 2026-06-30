import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Encore Drawing Matcher",
  description: "Match handwritten sketches to master drawings and fill JSON dimensions",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased" suppressHydrationWarning>
        <div className="min-h-screen">
          <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
            <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
              <div>
                <h1 className="text-xl font-semibold tracking-tight">Encore Drawing Matcher</h1>
                <p className="text-sm text-slate-400 mt-0.5">
                  Match handwritten sketches to master drawings and fill dimensions
                </p>
              </div>
              <nav className="flex items-center gap-1">
                <Link
                  href="/"
                  className="px-4 py-2 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors"
                >
                  Match
                </Link>
                <Link
                  href="/train"
                  className="px-4 py-2 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors"
                >
                  Train
                </Link>
                <Link
                  href="/batch"
                  className="px-4 py-2 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors"
                >
                  Batch
                </Link>
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
