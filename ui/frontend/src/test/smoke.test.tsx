import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom'; // ensure matchers are available

describe('Frontend Environment Smoke Test', () => {
  it('renders a simple boolean check', () => {
    render(<div />);
    expect(screen).toBeDefined();
    expect(true).toBe(true);
  });
});
