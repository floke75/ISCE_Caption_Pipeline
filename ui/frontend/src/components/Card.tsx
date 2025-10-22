import { PropsWithChildren, ReactNode } from "react";

interface CardProps {
  title: string;
  description?: string;
  actions?: ReactNode;
}

export function Card({ title, description, actions, children }: PropsWithChildren<CardProps>) {
  return (
    <section className="card">
      <header className="card__header">
        <div>
          <h2>{title}</h2>
          {description && <p className="card__description">{description}</p>}
        </div>
        {actions && <div className="card__actions">{actions}</div>}
      </header>
      <div className="card__body">{children}</div>
    </section>
  );
}

export default Card;
