import { PropsWithChildren } from "react";
import clsx from "clsx";

type Props = PropsWithChildren<{
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  className?: string;
}>;

export function SectionCard({ title, subtitle, actions, className, children }: Props) {
  return (
    <section
      className={clsx(
        "rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-900/40",
        className
      )}
    >
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
          {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
        </div>
        {actions && <div className="text-sm text-slate-300">{actions}</div>}
      </header>
      <div className="space-y-4 text-slate-200">{children}</div>
    </section>
  );
}
